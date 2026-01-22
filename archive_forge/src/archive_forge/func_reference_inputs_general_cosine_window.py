import unittest
from functools import partial
from itertools import product
from typing import Callable, List, Tuple
import numpy
import torch
from torch.testing._internal.common_dtype import floating_types
from torch.testing._internal.common_utils import TEST_SCIPY
from torch.testing._internal.opinfo.core import (
def reference_inputs_general_cosine_window(op_info, device, dtype, requires_grad, **kwargs):
    yield from sample_inputs_window(op_info, device, dtype, requires_grad, **kwargs)
    cases = ((8, {'a': [0.5, 0.5]}), (16, {'a': [0.46, 0.54]}), (32, {'a': [0.46, 0.23, 0.31]}), (64, {'a': [0.5]}), (128, {'a': [0.1, 0.8, 0.05, 0.05]}), (256, {'a': [0.2, 0.2, 0.2, 0.2, 0.2]}))
    for size, kw in cases:
        yield SampleInput(size, sym=False, **kw)
        yield SampleInput(size, sym=True, **kw)