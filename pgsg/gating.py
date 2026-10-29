"""Physics-Guided Safety Gating (PGSG).

Pixel-wise fusion of a conservative pre-trained model (f_origin, "safe") and a
domain-retrained model (f_retrain) driven by a compound risk map:

    R     = Norm(sigma^2 + D_res) * W_struct                  (Eq. 8)
    alpha = Sigmoid(gain * (R - tau) / tau)                   (Eq. 9)
    y     = alpha * mu + (1 - alpha) * f_origin               (Eq. 10)

where mu is the TTA mean of f_retrain. See README.md ("Implementation notes").
"""
import math

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

_SOBEL = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])


def gaussian_smooth(x, sigma):
    k = int(2 * math.ceil(2 * sigma) + 1)
    if k % 2 == 0:
        k += 1
    return TF.gaussian_blur(x, [k, k], [sigma, sigma])


def sobel_magnitude(x):
    """||grad x||_1 = |G_x| + |G_y| with Sobel kernels, averaged over RGB.

    Divided by 8 so that a unit-slope ramp has magnitude 1, which keeps the
    suppression coefficient lambda independent of the kernel's gain.
    """
    c = x.shape[1]
    kx = _SOBEL.to(x).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    ky = _SOBEL.t().to(x).reshape(1, 1, 3, 3).repeat(c, 1, 1, 1)
    xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
    grad = torch.abs(F.conv2d(xp, kx, groups=c)) + torch.abs(F.conv2d(xp, ky, groups=c))
    return grad.mean(dim=1, keepdim=True) / 8


def minmax(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


class PhysicsGuidedSafetyGater:
    """
    Args:
        tau: risk threshold on the normalized risk.
        gain: sharpness of the sigmoid transition (gamma in the paper).
        protection: structure-suppression sensitivity (lambda in Eq. 7).
    """

    def __init__(self, tau=0.07, gain=1000.0, protection=15.0):
        self.tau = tau
        self.gain = gain
        self.protection = protection

    def estimate_risk(self, tta_preds, y_safe):
        stacked = torch.stack(tta_preds)

        # (a) Epistemic uncertainty: variance across TTA views (Eq. 5)
        variance = torch.var(stacked, dim=0).mean(dim=1, keepdim=True)

        # (b) Reconstruction residual between retrained mean and safe baseline (Eq. 6)
        residual = torch.abs(stacked.mean(dim=0) - y_safe).mean(dim=1, keepdim=True)

        # (c) Structure-aware suppression: veto strong edges of the safe output (Eq. 7)
        w_struct = torch.exp(-self.protection * sobel_magnitude(y_safe))

        # Compound risk (Eq. 8): normalize the evidence, then apply the veto
        risk = minmax(variance + residual) * w_struct
        return gaussian_smooth(risk, sigma=1.0)

    def apply(self, y_safe, y_retrain, risk):
        # Soft gate (Eq. 9); tau-relative so that gain is comparable across thresholds
        alpha = torch.sigmoid(self.gain * (risk - self.tau) / (self.tau + 1e-8))
        # Feather the gate boundary to avoid seams between the two sources
        alpha = gaussian_smooth(alpha, sigma=3.0)
        # Fusion (Eq. 10)
        return alpha * y_retrain + (1 - alpha) * y_safe, alpha

    def __call__(self, tta_preds, y_safe):
        y_mean = torch.stack(tta_preds).mean(dim=0)
        risk = self.estimate_risk(tta_preds, y_safe)
        output, alpha = self.apply(y_safe, y_mean, risk)
        return output, alpha, risk
