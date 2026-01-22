from typing import TYPE_CHECKING
import torch
from . import allowed_functions
from .eval_frame import DisableContext, innermost_fn, RunOnlyContext
from .exc import IncorrectUsage
def forbid_in_graph(fn):
    """
    Customize which functions TorchDynamo will assert are not present while tracing.

    If you want a graph break on this function instead, use disallow_in_graph.
    TODO(voz): We now have allow_in_graph, disallow_in_graph, forbid_in_graph - some more robust
    documentation would not be amiss.
    """
    if isinstance(fn, (list, tuple)):
        return [forbid_in_graph(x) for x in fn]
    assert callable(fn), 'forbid_in_graph applies only to callables'
    fn._dynamo_forbidden = True
    return fn