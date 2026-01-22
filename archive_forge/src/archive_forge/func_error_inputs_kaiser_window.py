import unittest
from functools import partial
from itertools import product
from typing import Callable, List, Tuple
import numpy
import torch
from torch.testing._internal.common_dtype import floating_types
from torch.testing._internal.common_utils import TEST_SCIPY
from torch.testing._internal.opinfo.core import (
def error_inputs_kaiser_window(op_info, device, **kwargs):
    yield from error_inputs_window(op_info, device, beta=12, **kwargs)
    yield ErrorInput(SampleInput(3, beta=-1, dtype=torch.float32, device=device, **kwargs), error_type=ValueError, error_regex='beta must be non-negative, got: -1 instead.')