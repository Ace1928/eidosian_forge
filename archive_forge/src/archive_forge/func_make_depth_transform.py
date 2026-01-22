import argparse
import itertools
import math
from pathlib import Path
import requests
import torch
from PIL import Image
from torchvision import transforms
from transformers import Dinov2Config, DPTConfig, DPTForDepthEstimation, DPTImageProcessor
from transformers.utils import logging
def make_depth_transform() -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor(), lambda x: 255.0 * x[:3], transforms.Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)), CenterPadding(multiple=14)])