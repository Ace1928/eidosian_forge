from typing import TYPE_CHECKING
import torch
from . import allowed_functions
from .eval_frame import DisableContext, innermost_fn, RunOnlyContext
from .exc import IncorrectUsage
@forbid_in_graph
def maybe_mark_dynamic(t, index):
    """
    Mark a tensor as having a dynamic dim, but don't enforce it (i.e., if this
    dimension ends up getting specialized, don't error).
    """
    if isinstance(index, int):
        if not hasattr(t, '_dynamo_weak_dynamic_indices'):
            t._dynamo_weak_dynamic_indices = set()
        t._dynamo_weak_dynamic_indices.add(index)
        return
    assert isinstance(index, (list, tuple))
    for i in index:
        maybe_mark_dynamic(t, i)