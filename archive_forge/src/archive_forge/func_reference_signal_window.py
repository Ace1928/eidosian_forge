import unittest
from functools import partial
from itertools import product
from typing import Callable, List, Tuple
import numpy
import torch
from torch.testing._internal.common_dtype import floating_types
from torch.testing._internal.common_utils import TEST_SCIPY
from torch.testing._internal.opinfo.core import (
def reference_signal_window(fn: Callable):
    """Wrapper for scipy signal window references.

    Discards keyword arguments for window reference functions that don't have a matching signature with
    torch, e.g., gaussian window.
    """

    def _fn(*args, dtype=numpy.float64, device=None, layout=torch.strided, requires_grad=False, **kwargs):
        """The unused arguments are defined to disregard those values"""
        return fn(*args, **kwargs).astype(dtype)
    return _fn