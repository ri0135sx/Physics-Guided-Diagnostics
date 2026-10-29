"""Loaders for the five benchmarked architectures.

Network definitions are imported directly from the official repositories,
pinned as git submodules under third_party/ (NAFNet is re-declared in
pgsg/archs/nafnet.py to avoid pulling in its BasicSR package).
"""
import importlib.util
import os
import warnings

import torch

from .archs.nafnet import NAFNet

THIRD_PARTY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party")

ARCH_SOURCES = {
    "HINet":     ("HINet/basicsr/models/archs/hinet_arch.py", "HINet"),
    "PReNet":    ("PReNet/networks.py", "PReNet"),
    "Restormer": ("Restormer/basicsr/models/archs/restormer_arch.py", "Restormer"),
    "MPRNet":    ("MPRNet/Deraining/MPRNet.py", "MPRNet"),
}

ARCH_KWARGS = {
    "HINet":     dict(in_chn=3, wf=64, depth=5),
    "PReNet":    dict(recurrent_iter=6, use_GPU=True),
    "Restormer": dict(inp_channels=3, out_channels=3, dim=48, num_blocks=[4, 6, 6, 8],
                      num_refinement_blocks=4, heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                      bias=False, LayerNorm_type="WithBias", dual_pixel_task=False),
    "MPRNet":    dict(in_c=3, out_c=3, n_feat=40, scale_unetfeats=20, scale_orsnetfeats=16),
}

# NAFNet checkpoints: GoPro width-32 (original) and width-64 SIDD-style (retrained)
NAFNET_VARIANTS = [
    dict(width=32, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1]),
    dict(width=64, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]),
]


class FinalOutput(torch.nn.Module):
    """Expose only the final restored image of multi-stage networks.

    MPRNet returns [stage3, stage2, stage1], HINet [stage1, stage2] and
    PReNet (final, [per-iteration outputs]), so the final stage is not
    always the last element.
    """

    SELECT = {"MPRNet": 0, "HINet": -1, "PReNet": 0}

    def __init__(self, model, arch):
        super().__init__()
        self.model = model
        self.index = self.SELECT.get(arch)

    def forward(self, x):
        out = self.model(x)
        return out if self.index is None else out[self.index]


def _import_class(rel_path, class_name):
    path = os.path.join(THIRD_PARTY, rel_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; run `git submodule update --init`")
    spec = importlib.util.spec_from_file_location(f"pgsg_third_party_{class_name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def _strip_prefixes(key, prefixes=("module.", "net.")):
    # Only strip leading wrappers: a plain replace() would corrupt keys such as "stage3_orsnet.orb1"
    for p in prefixes:
        if key.startswith(p):
            key = key[len(p):]
    return key


def _read_state_dict(weight_path):
    ckpt = torch.load(weight_path, map_location="cpu")
    state = ckpt.get("params", ckpt.get("state_dict", ckpt))
    return {_strip_prefixes(k): v for k, v in state.items()}


def _load_nafnet(weight_path):
    state = _read_state_dict(weight_path)
    variants = NAFNET_VARIANTS if "32" in os.path.basename(weight_path) else NAFNET_VARIANTS[::-1]
    for cfg in variants:
        model = NAFNet(**cfg)
        try:
            model.load_state_dict(state, strict=True)
            return model
        except RuntimeError:
            continue
    raise RuntimeError(f"{weight_path} matches no known NAFNet configuration")


def load_model(arch, weight_path, device="cuda"):
    if arch == "NAFNet":
        model = _load_nafnet(weight_path)
    else:
        model = _import_class(*ARCH_SOURCES[arch])(**ARCH_KWARGS[arch])
        missing, _ = model.load_state_dict(_read_state_dict(weight_path), strict=False)
        if missing:
            warnings.warn(f"{arch}: {len(missing)} missing keys when loading {weight_path}")
    return FinalOutput(model, arch).eval().to(device)
