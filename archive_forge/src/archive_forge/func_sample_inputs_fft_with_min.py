import unittest
from functools import partial
from typing import List
import numpy as np
import torch
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import SM53OrLater
from torch.testing._internal.common_device_type import precisionOverride
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_utils import TEST_SCIPY, TEST_WITH_ROCM
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.opinfo.refs import (
def sample_inputs_fft_with_min(op_info, device, dtype, requires_grad=False, *, min_size, **kwargs):
    yield from sample_inputs_spectral_ops(op_info, device, dtype, requires_grad, **kwargs)
    if TEST_WITH_ROCM:
        return
    a = make_tensor(min_size, dtype=dtype, device=device, requires_grad=requires_grad)
    yield SampleInput(a)