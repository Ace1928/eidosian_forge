from typing import List
import torch
def integral_types_and(*dtypes):
    return _integral_types + _validate_dtypes(*dtypes)