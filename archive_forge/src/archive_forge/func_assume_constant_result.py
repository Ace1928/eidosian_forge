from typing import TYPE_CHECKING
import torch
from . import allowed_functions
from .eval_frame import DisableContext, innermost_fn, RunOnlyContext
from .exc import IncorrectUsage
def assume_constant_result(fn):
    fn._dynamo_marked_constant = True
    return fn