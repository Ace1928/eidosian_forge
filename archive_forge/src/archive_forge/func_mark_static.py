from typing import TYPE_CHECKING
import torch
from . import allowed_functions
from .eval_frame import DisableContext, innermost_fn, RunOnlyContext
from .exc import IncorrectUsage
@forbid_in_graph
def mark_static(t, index=None):
    """
    Mark a tensor as having a static dim.

    This will prevent us from attempting to compile it dynamically
    when dynamic=True; this can improve trace-time performance.

    This has lower precedence than mark_dynamic.
    """
    if isinstance(index, int):
        if not hasattr(t, '_dynamo_static_indices'):
            t._dynamo_static_indices = set()
        t._dynamo_static_indices.add(index)
    elif index is None:
        for i in range(t.dim()):
            mark_static(t, i)
    else:
        assert isinstance(index, (list, tuple))
        for i in index:
            mark_static(t, i)