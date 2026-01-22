import collections
import functools
import torch
from typing import Any
def custom_bwd(bwd):
    """Create a helper decorator for backward methods of custom autograd functions.

    Autograd functions are subclasses of :class:`torch.autograd.Function`.
    Ensures that ``backward`` executes with the same autocast state as ``forward``.
    See the :ref:`example page<amp-custom-examples>` for more detail.
    """

    @functools.wraps(bwd)
    def decorate_bwd(*args, **kwargs):
        with autocast(enabled=args[0]._fwd_used_autocast, dtype=args[0]._dtype):
            return bwd(*args, **kwargs)
    return decorate_bwd