import os
import pathlib
import torch
from torch.jit._recursive import wrap_cpp_module
from torch.serialization import validate_cuda_device
def jit_module_from_flatbuffer(f):
    if isinstance(f, (str, pathlib.Path)):
        f = str(f)
        return wrap_cpp_module(torch._C._load_jit_module_from_file(f))
    else:
        return wrap_cpp_module(torch._C._load_jit_module_from_bytes(f.read()))