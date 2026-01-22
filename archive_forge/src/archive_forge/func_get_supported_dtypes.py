import collections
import warnings
from functools import partial, wraps
from typing import Sequence
import numpy as np
import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_utils import torch_to_numpy_dtype_dict
def get_supported_dtypes(op, sample_inputs_fn, device_type):
    assert device_type in ['cpu', 'cuda']
    if not TEST_CUDA and device_type == 'cuda':
        warnings.warn('WARNING: CUDA is not available, empty_dtypes dispatch will be returned!')
        return _dynamic_dispatch_dtypes(())
    supported_dtypes = set()
    for dtype in all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half):
        try:
            samples = sample_inputs_fn(op, device_type, dtype, False)
        except RuntimeError:
            warnings.warn(f'WARNING: Unable to generate sample for device:{device_type} and dtype:{dtype}')
            continue
        supported = True
        for sample in samples:
            try:
                op(sample.input, *sample.args, **sample.kwargs)
            except RuntimeError as re:
                supported = False
                break
        if supported:
            supported_dtypes.add(dtype)
    return _dynamic_dispatch_dtypes(supported_dtypes)