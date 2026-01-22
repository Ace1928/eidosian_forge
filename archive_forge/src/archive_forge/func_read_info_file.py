import os
from typing import Any, Callable, List, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image
from .utils import download_url
from .vision import VisionDataset
def read_info_file(data_dir: str, info_file: str) -> torch.Tensor:
    """Return a Tensor containing the list of labels
    Read the file and keep only the ID of the 3D point.
    """
    with open(os.path.join(data_dir, info_file)) as f:
        labels = [int(line.split()[0]) for line in f]
    return torch.LongTensor(labels)