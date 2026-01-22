import itertools
import warnings
from typing import Protocol
import torch
from ..parameter import is_lazy
def has_uninitialized_params(self: _LazyProtocol):
    """Check if a module has parameters that are not initialized."""
    params = self._parameters.values()
    buffers = self._buffers.values()
    for param in itertools.chain(params, buffers):
        if is_lazy(param):
            return True
    return False