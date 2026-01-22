import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
def check_cat_no_zero_dim(tensors: List[List[int]]):
    for tensor in tensors:
        assert len(tensor) > 0