from typing import TYPE_CHECKING
import torch
from . import allowed_functions
from .eval_frame import DisableContext, innermost_fn, RunOnlyContext
from .exc import IncorrectUsage
@_disallow_in_graph_helper(throw_if_not_allowed=False)
def graph_break():
    """Force a graph break"""
    pass