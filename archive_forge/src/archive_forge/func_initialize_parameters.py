import itertools
import warnings
from typing import Protocol
import torch
from ..parameter import is_lazy
def initialize_parameters(self: _LazyProtocol, *args, **kwargs):
    """Initialize parameters according to the input batch properties.

        This adds an interface to isolate parameter initialization from the
        forward pass when doing parameter shape inference.
        """
    raise NotImplementedError(f'initialize_parameters is not implemented for {self.__class__.__name__}')