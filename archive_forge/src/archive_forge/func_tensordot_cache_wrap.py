import contextlib
import functools
import numbers
import threading
from collections import Counter, defaultdict
from .parser import alpha_canonicalize, parse_einsum_input
def tensordot_cache_wrap(tensordot):
    """Decorates a ``tensordot()`` implementation to be memoized inside a
    :func:`shared_intermediates` context.
    """

    @functools.wraps(tensordot)
    def cached_tensordot(x, y, axes=2, backend='numpy'):
        if not currently_sharing():
            return tensordot(x, y, axes, backend=backend)
        _save_tensors(x, y)
        if isinstance(axes, numbers.Number):
            axes = (list(range(len(x.shape)))[len(x.shape) - axes:], list(range(len(y.shape)))[:axes])
        axes = (tuple(axes[0]), tuple(axes[1]))
        key = ('tensordot', backend, id(x), id(y), axes)
        return _memoize(key, tensordot, x, y, axes, backend=backend)
    return cached_tensordot