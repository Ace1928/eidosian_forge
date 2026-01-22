import contextlib
import functools
import numbers
import threading
from collections import Counter, defaultdict
from .parser import alpha_canonicalize, parse_einsum_input
def einsum_cache_wrap(einsum):
    """Decorates an ``einsum()`` implementation to be memoized inside a
    :func:`shared_intermediates` context.
    """

    @functools.wraps(einsum)
    def cached_einsum(*args, **kwargs):
        if not currently_sharing():
            return einsum(*args, **kwargs)
        backend = kwargs.pop('backend', 'numpy')
        equation = args[0]
        inputs, output, operands = parse_einsum_input(args)
        inputs = inputs.split(',')
        _save_tensors(*operands)
        canonical = sorted(zip(inputs, map(id, operands)), key=lambda x: x[1])
        canonical_ids = tuple((id_ for _, id_ in canonical))
        canonical_inputs = ','.join((input_ for input_, _ in canonical))
        canonical_equation = alpha_canonicalize(canonical_inputs + '->' + output)
        key = ('einsum', backend, canonical_equation, canonical_ids)
        return _memoize(key, einsum, equation, *operands, backend=backend)
    return cached_einsum