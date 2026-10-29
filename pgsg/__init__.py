"""Physics-Guided Safety Gating (PGSG) for raindrop removal."""
import os

# Avoid fragmentation so that 1080p full-image inference fits on 12 GB GPUs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from .gating import PhysicsGuidedSafetyGater
from .inference import predict, predict_tta
from .models import load_model
from .configs import GATING_CONFIGS, WEIGHTS, ARCHS
