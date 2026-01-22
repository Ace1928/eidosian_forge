import os
import pathlib
import torch
from torch.jit._recursive import wrap_cpp_module
from torch.serialization import validate_cuda_device
def validate_map_location(map_location=None):
    if isinstance(map_location, str):
        map_location = torch.device(map_location)
    elif not (map_location is None or isinstance(map_location, torch.device)):
        raise ValueError('map_location should be either None, string or torch.device, but got type: ' + str(type(map_location)))
    if str(map_location).startswith('cuda'):
        validate_cuda_device(map_location)
    return map_location