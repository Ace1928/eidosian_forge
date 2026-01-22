from functools import lru_cache as _lru_cache
import torch
from ...library import Library as _Library
@_lru_cache
def is_macos13_or_newer(minor: int=0) -> bool:
    """Return a bool indicating whether MPS is running on MacOS 13 or newer."""
    return torch._C._mps_is_on_macos_13_or_newer(minor)