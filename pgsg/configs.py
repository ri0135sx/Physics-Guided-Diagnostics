# Gating parameters.
#   tau        : threshold on the min-max normalized risk; shared by all architectures
#   gain       : sigmoid sharpness (gamma)
#   protection : structure-suppression sensitivity (lambda)
TAU = 0.07

GATING_CONFIGS = {
    "NAFNet":    {"tau": TAU, "gain": 1000.0, "protection": 20.0},
    "PReNet":    {"tau": TAU, "gain": 800.0,  "protection": 15.0},
    "HINet":     {"tau": TAU, "gain": 1000.0, "protection": 15.0},
    "MPRNet":    {"tau": TAU, "gain": 1000.0, "protection": 15.0},
    "Restormer": {"tau": TAU, "gain": 1500.0, "protection": 10.0},
}

# (original / pre-trained, retrained on RainDropClarity) checkpoint file names
# expected under --weights_dir. Original weights come from each official repo.
WEIGHTS = {
    "NAFNet":    ("NAFNet-GoPro-width32.pth", "NAFNet.pth"),
    "HINet":     ("HINet-Rain13k.pth",        "HINet.pth"),
    "PReNet":    ("net_latest.pth",           "PReNet.pth"),
    "Restormer": ("deraining.pth",            "restormer_drop.pth"),
    "MPRNet":    ("model_raindrop.pth",       "MPR_drop.pth"),
}

ARCHS = list(WEIGHTS)
