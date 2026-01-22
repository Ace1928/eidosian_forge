from collections import namedtuple
import torch
from . import _casting_dicts as _cd
def python_type_for_torch(dtyp):
    """Get a python scalar type a torch dtype"""
    if dtyp.is_floating_point:
        typ = float
    elif dtyp.is_complex:
        typ = complex
    elif dtyp == torch.bool:
        typ = bool
    else:
        typ = int
    return typ