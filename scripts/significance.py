"""Wilcoxon signed-rank test of PGSG vs. Retrained + TTA on per-image metrics.

Example:
    python scripts/significance.py results/per_image_metrics.csv
"""
import argparse

import pandas as pd
from scipy.stats import wilcoxon


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="output of scripts/evaluate.py")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    for arch, g in df.groupby("Arch", sort=False):
        line = f"{arch:<10} N={len(g):<4}"
        for metric in ("PSNR", "SSIM"):
            _, p_val = wilcoxon(g[f"PGSG_{metric}"], g[f"TTA_{metric}"])
            gain = (g[f"PGSG_{metric}"] - g[f"TTA_{metric}"]).mean()
            line += f"  {metric}: gain={gain:+.4f} p={p_val:.2e}"
        print(line)


if __name__ == "__main__":
    main()
