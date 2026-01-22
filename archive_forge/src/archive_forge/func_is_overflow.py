import inspect
import warnings
import torch
from .state import AcceleratorState, GradientState
from .utils import DistributedType, honor_type, is_torch_xla_available
@property
def is_overflow(self):
    """Whether or not the optimizer step was done, or skipped because of gradient overflow."""
    warnings.warn('The `is_overflow` property is deprecated and will be removed in version 1.0 of Accelerate use `optimizer.step_was_skipped` instead.', FutureWarning)
    return self._is_overflow