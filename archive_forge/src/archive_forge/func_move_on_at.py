from __future__ import annotations
import math
from contextlib import AbstractContextManager, contextmanager
from typing import TYPE_CHECKING
import trio
def move_on_at(deadline: float) -> trio.CancelScope:
    """Use as a context manager to create a cancel scope with the given
    absolute deadline.

    Args:
      deadline (float): The deadline.

    Raises:
      ValueError: if deadline is NaN.

    """
    if math.isnan(deadline):
        raise ValueError('deadline must not be NaN')
    return trio.CancelScope(deadline=deadline)