from __future__ import absolute_import
from builtins import range
import autograd.numpy as np
from autograd.misc import flatten
from autograd.wrap_util import wraps
def unflatten_optimizer(optimize):
    """Takes an optimizer that operates on flat 1D numpy arrays and returns a
    wrapped version that handles trees of nested containers (lists/tuples/dicts)
    with arrays/scalars at the leaves."""

    @wraps(optimize)
    def _optimize(grad, x0, callback=None, *args, **kwargs):
        _x0, unflatten = flatten(x0)
        _grad = lambda x, i: flatten(grad(unflatten(x), i))[0]
        if callback:
            _callback = lambda x, i, g: callback(unflatten(x), i, unflatten(g))
        else:
            _callback = None
        return unflatten(optimize(_grad, _x0, _callback, *args, **kwargs))
    return _optimize