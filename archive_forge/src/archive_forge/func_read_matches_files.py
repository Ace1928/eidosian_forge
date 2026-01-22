import os
from typing import Any, Callable, List, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image
from .utils import download_url
from .vision import VisionDataset
def read_matches_files(data_dir: str, matches_file: str) -> torch.Tensor:
    """Return a Tensor containing the ground truth matches
    Read the file and keep only 3D point ID.
    Matches are represented with a 1, non matches with a 0.
    """
    matches = []
    with open(os.path.join(data_dir, matches_file)) as f:
        for line in f:
            line_split = line.split()
            matches.append([int(line_split[0]), int(line_split[3]), int(line_split[1] == line_split[4])])
    return torch.LongTensor(matches)