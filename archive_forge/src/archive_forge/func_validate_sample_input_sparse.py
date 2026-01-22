import os
import torch
from torch.testing import make_tensor  # noqa: F401
from torch.testing._internal.opinfo.core import (  # noqa: F401
def validate_sample_input_sparse(op_info, sample, check_validate=False):
    """Return the specified sample when it is valid and supported by the
    operation. Otherwise, return the sample as ErrorInput instance.

    When check_validate is True, the result is validated against
    calling the op on the sample.
    """
    if isinstance(op_info, ReductionOpInfo):
        return _validate_sample_input_sparse_reduction(op_info, sample, check_validate=check_validate)
    elif isinstance(op_info, BinaryUfuncInfo):
        return _validate_sample_input_sparse_elementwise_binary_operation(op_info, sample, check_validate=check_validate)
    else:
        return _validate_sample_input_sparse_default(op_info, sample, check_validate=check_validate)