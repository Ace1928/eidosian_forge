import os
import pathlib
import torch
from torch.jit._serialization import validate_map_location
def find_method(self, method_name):
    return self._c.find_method(method_name)