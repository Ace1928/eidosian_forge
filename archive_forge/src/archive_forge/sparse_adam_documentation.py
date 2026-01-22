import torch
from . import _functional as F
from .optimizer import Optimizer, _maximize_doc
Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        