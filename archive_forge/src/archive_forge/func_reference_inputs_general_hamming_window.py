import unittest
from functools import partial
from itertools import product
from typing import Callable, List, Tuple
import numpy
import torch
from torch.testing._internal.common_dtype import floating_types
from torch.testing._internal.common_utils import TEST_SCIPY
from torch.testing._internal.opinfo.core import (
def reference_inputs_general_hamming_window(op_info, device, dtype, requires_grad, **kwargs):
    yield from sample_inputs_window(op_info, device, dtype, requires_grad, **kwargs)
    cases = ((8, {'alpha': 0.54}), (16, {'alpha': 0.5}), (32, {'alpha': 0.23}), (64, {'alpha': 0.8}), (128, {'alpha': 0.9}), (256, {'alpha': 0.05}))
    for size, kw in cases:
        yield SampleInput(size, sym=False, **kw)
        yield SampleInput(size, sym=True, **kw)