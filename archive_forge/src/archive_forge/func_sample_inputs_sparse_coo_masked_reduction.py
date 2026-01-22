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
def sample_inputs_sparse_coo_masked_reduction(op_info, device, dtype, requires_grad, **kwargs):
    """Sample inputs for masked reduction operators that support inputs
    with sparse coo layouts.
    """
    if op_info.supports_sparse:
        op_name = op_info.name.replace('masked.', '')
        for sample_input in sample_inputs_masked_reduction(op_info, device, dtype, requires_grad, **kwargs):
            mask = sample_input.kwargs.get('mask')
            if mask is not None:
                sample_input_kwargs = sample_input.kwargs.copy()
                sample_input_kwargs.update(mask=mask.to_sparse())
                yield SampleInput(sample_input.input.to_sparse(), args=sample_input.args, kwargs=sample_input_kwargs)
            else:
                if op_name in {'prod', 'amax', 'amin'}:
                    continue
                yield SampleInput(sample_input.input.to_sparse(), args=sample_input.args, kwargs=sample_input.kwargs)