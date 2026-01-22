import builtins
import torch
from . import _dtypes_impl
def sctype_from_torch_dtype(torch_dtype):
    return _torch_dtypes[torch_dtype]