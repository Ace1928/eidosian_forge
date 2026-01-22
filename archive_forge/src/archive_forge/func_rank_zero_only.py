import os
import warnings
from functools import partial, wraps
from typing import Any, Callable
from torchmetrics import _logger as log
def rank_zero_only(fn: Callable) -> Callable:
    """Call a function only on rank 0 in distributed settings.

    Meant to be used as an decorator.

    """

    @wraps(fn)
    def wrapped_fn(*args: Any, **kwargs: Any) -> Any:
        if rank_zero_only.rank == 0:
            return fn(*args, **kwargs)
        return None
    return wrapped_fn