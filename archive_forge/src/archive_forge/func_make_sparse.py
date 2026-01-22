import math
from torch import Tensor
from typing import List
from .adadelta import adadelta  # type: ignore[attr-defined] # noqa: F401
from .adagrad import adagrad, _make_sparse  # type: ignore[attr-defined] # noqa: F401
from .adam import adam  # type: ignore[attr-defined] # noqa: F401
from .adamw import adamw  # type: ignore[attr-defined] # noqa: F401
from .adamax import adamax  # type: ignore[attr-defined] # noqa: F401
from .asgd import asgd  # type: ignore[attr-defined] # noqa: F401
from .nadam import nadam  # type: ignore[attr-defined] # noqa: F401
from .radam import radam  # type: ignore[attr-defined] # noqa: F401
from .rmsprop import rmsprop  # type: ignore[attr-defined] # noqa: F401
from .rprop import rprop  # type: ignore[attr-defined] # noqa: F401
from .sgd import sgd  # type: ignore[attr-defined] # noqa: F401
def make_sparse(values):
    constructor = grad.new
    if grad_indices.dim() == 0 or values.dim() == 0:
        return constructor().resize_as_(grad)
    return constructor(grad_indices, values, size)