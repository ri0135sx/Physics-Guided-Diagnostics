"""Apply PGSG to a single image and save the output, gate (alpha) and risk map.

Example:
    python scripts/run_pgsg.py --arch HINet \
        --origin weights/HINet-Rain13k.pth --retrained weights/HINet.pth \
        --input rainy.png --out_dir outputs/
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pgsg import ARCHS, GATING_CONFIGS, PhysicsGuidedSafetyGater, load_model, predict, predict_tta


def to_bgr(t):
    img = (t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def to_heatmap(t):
    img = (t.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(img, cv2.COLORMAP_JET)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=ARCHS, required=True)
    p.add_argument("--origin", required=True, help="pre-trained (safe) checkpoint")
    p.add_argument("--retrained", required=True, help="checkpoint retrained on raindrop data")
    p.add_argument("--input", required=True)
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--tau", type=float, help="override the architecture default")
    args = p.parse_args()

    cfg = dict(GATING_CONFIGS[args.arch])
    if args.tau is not None:
        cfg["tau"] = args.tau
    gater = PhysicsGuidedSafetyGater(**cfg)

    m_origin = load_model(args.arch, args.origin)
    m_retrain = load_model(args.arch, args.retrained)

    img = cv2.cvtColor(cv2.imread(args.input), cv2.COLOR_BGR2RGB)
    x = TF.to_tensor(img).unsqueeze(0).cuda()
    with torch.no_grad():
        y_origin = predict(m_origin, x)
        output, alpha, risk = gater(predict_tta(m_retrain, x), y_origin)

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input))[0]
    cv2.imwrite(os.path.join(args.out_dir, f"{stem}_pgsg.png"), to_bgr(output))
    cv2.imwrite(os.path.join(args.out_dir, f"{stem}_alpha.png"), to_heatmap(alpha))
    cv2.imwrite(os.path.join(args.out_dir, f"{stem}_risk.png"), to_heatmap(risk))
    print(f"saved to {args.out_dir}/ (mean alpha = {alpha.mean().item():.3f})")


if __name__ == "__main__":
    main()
