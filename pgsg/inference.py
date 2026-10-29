import warnings

import torch
import torch.nn.functional as F


def _pad_to_multiple(x, factor=64):
    h, w = x.shape[2], x.shape[3]
    ph, pw = (factor - h % factor) % factor, (factor - w % factor) % factor
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x


def _forward(model, x, precision):
    # fp16 autocast keeps full-image inference (and thus Restormer's global
    # attention) within ~10 GB at 1080p; fp32 matches the paper's 24 GB setup.
    with torch.autocast("cuda", dtype=torch.float16, enabled=precision == "fp16"):
        return model(x).float()


@torch.no_grad()
def predict_tiled(model, x, precision="fp32", tile_size=512, tile_overlap=64):
    _, _, h, w = x.shape
    output = torch.zeros_like(x)
    count = torch.zeros_like(x)
    stride = tile_size - tile_overlap
    for y in range(0, h, stride):
        for xx in range(0, w, stride):
            y_s, y_e = y, min(y + tile_size, h)
            x_s, x_e = xx, min(xx + tile_size, w)
            if y_e - y_s < tile_size:
                y_s = max(0, y_e - tile_size)
            if x_e - x_s < tile_size:
                x_s = max(0, x_e - tile_size)
            output[..., y_s:y_e, x_s:x_e] += _forward(model, x[..., y_s:y_e, x_s:x_e], precision)
            count[..., y_s:y_e, x_s:x_e] += 1.0
    return output.div_(count)


@torch.no_grad()
def predict(model, x, precision="fp32"):
    """Full-image inference with reflect padding; falls back to tiling on OOM.

    precision: "fp32" (paper setting, ~24 GB GPU for Restormer at 1080p) or
    "fp16" (autocast, fits 12 GB GPUs while keeping full-image inference).
    """
    h, w = x.shape[2], x.shape[3]
    out = None
    try:
        out = _forward(model, _pad_to_multiple(x), precision)[:, :, :h, :w]
    except torch.cuda.OutOfMemoryError:
        pass
    if out is None:
        # Tile outside the except block so the failed attempt's tensors are freed
        warnings.warn("out of GPU memory: falling back to tiled inference, which changes the "
                      "output of global-attention models such as Restormer; try --precision fp16")
        torch.cuda.empty_cache()
        out = predict_tiled(model, x, precision)
    return torch.clamp(out, 0, 1)


@torch.no_grad()
def predict_tta(model, x, precision="fp32"):
    """K = 3 views: identity, horizontal flip, vertical flip (Eq. 5)."""
    return [
        predict(model, x, precision),
        torch.flip(predict(model, torch.flip(x, [3]), precision), [3]),
        torch.flip(predict(model, torch.flip(x, [2]), precision), [2]),
    ]
