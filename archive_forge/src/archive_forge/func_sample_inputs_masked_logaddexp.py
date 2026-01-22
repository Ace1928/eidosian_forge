import unittest
from collections.abc import Sequence
from functools import partial
from typing import List
import numpy as np
import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import tol, toleranceOverride
from torch.testing._internal.common_dtype import (
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.opinfo.utils import prod_numpy, reference_reduction_numpy
def sample_inputs_masked_logaddexp(op_info, device, dtype, requires_grad, **kwargs):
    """Sample inputs for masked logaddexp."""
    shapes = [(S,), (S, S), (S, M, S)]
    input_mask_lists = [list(_generate_masked_op_mask(shape, device, **kwargs)) for shape in shapes]
    other_mask_lists = [list(_generate_masked_op_mask(shape, device, **kwargs)) for shape in shapes]
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    for shape, input_masks, other_masks in zip(shapes, input_mask_lists, other_mask_lists):
        for input_mask, other_mask in zip(input_masks, other_masks):
            yield SampleInput(make_arg(shape), make_arg(shape), input_mask=input_mask, other_mask=other_mask)