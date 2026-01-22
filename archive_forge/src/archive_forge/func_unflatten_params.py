from contextlib import contextmanager
from itertools import chain
import typing
from typing import (
import torch
from torch import Tensor
import torch.nn as nn
from fairscale.internal.state_dict import replace_by_prefix_
@contextmanager
def unflatten_params(self, flat_params: Optional[List[Tensor]]=None) -> Generator:
    """
        Unflatten params. If the current instance is already unflattened, then
        it will remain unflattened after the context manager exits.

        Args:
            flat_params (List[Tensor], Optional):
                flat params to use for unflattening.
                If provided, the current instance must be in a flattened state
                at the start of the context manager. The provided Tensor must be
                appropriately sized and will only be used within the context
                manager. After the context manager exits, we will revert to
                using ``self.flat_params``
                Default: None.
        """
    assert flat_params is None or self.is_flattened, 'Unflattening with external flat_param requires current instance to be flattened'
    orig_flattened = self.is_flattened
    if orig_flattened:
        orig_flat_params = self.flat_params
        self._unflatten_params(cast(Optional[List[Optional[Tensor]]], flat_params))
    try:
        yield
    finally:
        if orig_flattened:
            self._flatten_params(orig_flat_params)