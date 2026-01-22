import os
from collections import namedtuple
from typing import Any
import torch
from .grad_mode import _DecoratorContextManager
def exit_dual_level(*, level=None):
    """Exit a forward grad level.

    This function deletes all the gradients associated with this
    level. Only deleting the latest entered level is allowed.

    This function also updates the current level that is used by default
    by the other functions in this API.
    """
    global _current_level
    if level is None:
        level = _current_level
    if level != _current_level:
        raise RuntimeError('Trying to exit a forward AD level that was not the last one that was created. This is not supported.')
    torch._C._exit_dual_level(level=level)
    _current_level = level - 1