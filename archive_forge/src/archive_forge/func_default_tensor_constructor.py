import functools
import itertools as it
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
@staticmethod
def default_tensor_constructor(size, dtype, **kwargs):
    if dtype.is_floating_point or dtype.is_complex:
        return torch.rand(size=size, dtype=dtype, device='cpu')
    else:
        return torch.randint(1, 127, size=size, dtype=dtype, device='cpu')