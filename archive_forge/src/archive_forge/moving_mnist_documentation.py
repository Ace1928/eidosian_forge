import os.path
from typing import Callable, Optional
import numpy as np
import torch
from torchvision.datasets.utils import download_url, verify_str_arg
from torchvision.datasets.vision import VisionDataset

        Args:
            index (int): Index
        Returns:
            torch.Tensor: Video frames (torch Tensor[T, C, H, W]). The `T` is the number of frames.
        