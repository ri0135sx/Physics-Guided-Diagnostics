import torch
import torch.nn.functional as F


def _pad_to_multiple(x, factor=64):
    h, w = x.shape[2], x.shape[3]
    ph, pw = (factor - h % factor) % factor, (factor - w % factor) % factor
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x


@torch.no_grad()
def predict_tiled(model, x, tile_size=512, tile_overlap=64):
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
            output[..., y_s:y_e, x_s:x_e] += model(x[..., y_s:y_e, x_s:x_e])
            count[..., y_s:y_e, x_s:x_e] += 1.0
    return output.div_(count)


@torch.no_grad()
def predict(model, x):
    """Full-image inference with reflect padding; falls back to tiling on OOM."""
    h, w = x.shape[2], x.shape[3]
    out = None
    try:
        out = model(_pad_to_multiple(x))[:, :, :h, :w]
    except (RuntimeError, torch.cuda.OutOfMemoryError):
        pass
    if out is None:
        # Tile outside the except block so the failed attempt's tensors are freed
        torch.cuda.empty_cache()
        out = predict_tiled(model, x)
    return torch.clamp(out, 0, 1)


@torch.no_grad()
def predict_tta(model, x):
    """K = 3 views: identity, horizontal flip, vertical flip (Eq. 5)."""
    return [
        predict(model, x),
        torch.flip(predict(model, torch.flip(x, [3])), [3]),
        torch.flip(predict(model, torch.flip(x, [2])), [2]),
    ]
