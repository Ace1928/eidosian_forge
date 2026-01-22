import os
import torch
from torch.testing import make_tensor  # noqa: F401
from torch.testing._internal.opinfo.core import (  # noqa: F401
def error_inputs_sparse_reduction_sum(op_info, device, layout, **kwargs):
    """Error inputs for sum on sparse tensors."""
    dtype = torch.float64
    requires_grad = False
    yield from _error_inputs_sparse(_maybe_failing_sample_inputs_sparse_reduction_sum, _validate_sample_input_sparse_reduction, op_info, device, dtype, requires_grad, layout, **kwargs)