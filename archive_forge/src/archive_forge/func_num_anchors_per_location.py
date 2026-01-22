import math
from typing import List, Optional
import torch
from torch import nn, Tensor
from .image_list import ImageList
def num_anchors_per_location(self) -> List[int]:
    return [2 + 2 * len(r) for r in self.aspect_ratios]