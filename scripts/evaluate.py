"""Benchmark Original / Retrained / Retrained+TTA / PGSG for each architecture (Table 1).

Expects paired images `Input_*.png` / `Target_*.png` in --data_dir and the
checkpoints listed in pgsg/configs.py under --weights_dir.

Example:
    python scripts/evaluate.py --data_dir data/test --weights_dir weights --out_dir results
"""
import argparse
import gc
import glob
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pgsg import ARCHS, GATING_CONFIGS, WEIGHTS, PhysicsGuidedSafetyGater, load_model, predict, predict_tta

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


def metrics(t, target_rgb):
    img = (t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    psnr = peak_signal_noise_ratio(target_rgb, img, data_range=255)
    ssim = structural_similarity(target_rgb, img, data_range=255, channel_axis=-1)
    return psnr, ssim


def evaluate_arch(arch, files, weights_dir, precision):
    origin_name, retrained_name = WEIGHTS[arch]
    m_origin = load_model(arch, os.path.join(weights_dir, origin_name))
    m_retrain = load_model(arch, os.path.join(weights_dir, retrained_name))
    gater = PhysicsGuidedSafetyGater(**GATING_CONFIGS[arch])

    rows = []
    for path in tqdm(files, desc=arch):
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(cv2.imread(path.replace("Input_", "Target_")), cv2.COLOR_BGR2RGB)
        x = TF.to_tensor(img).unsqueeze(0).cuda()

        with torch.no_grad():
            y_origin = predict(m_origin, x, precision)
            tta = predict_tta(m_retrain, x, precision)
            y_mean = torch.stack(tta).mean(dim=0)
            y_pgsg, alpha, _ = gater(tta, y_origin)

        row = {"Arch": arch, "File": os.path.basename(path)}
        for name, y in [("Origin", y_origin), ("Retrained", tta[0]), ("TTA", y_mean), ("PGSG", y_pgsg)]:
            row[f"{name}_PSNR"], row[f"{name}_SSIM"] = metrics(y, target)
        row["Alpha_Mean"] = alpha.mean().item()
        rows.append(row)

    del m_origin, m_retrain
    torch.cuda.empty_cache()
    gc.collect()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--weights_dir", default="weights")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--archs", nargs="+", default=ARCHS, choices=ARCHS)
    p.add_argument("--limit", type=int, help="evaluate only the first N images")
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp32",
                   help="fp32: paper setting (24 GB GPU); fp16: for 12 GB GPUs")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "Input_*.png")))[: args.limit]
    if not files:
        sys.exit(f"no Input_*.png found in {args.data_dir}")

    rows = []
    for arch in args.archs:
        rows += evaluate_arch(arch, files, args.weights_dir, args.precision)

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, "per_image_metrics.csv")
    df.to_csv(csv_path, index=False)

    cols = [f"{m}_{k}" for k in ("PSNR", "SSIM") for m in ("Origin", "Retrained", "TTA", "PGSG")]
    print(df.groupby("Arch", sort=False)[cols].mean().round(3).to_string())
    print(f"\nper-image results: {csv_path}")


if __name__ == "__main__":
    main()
