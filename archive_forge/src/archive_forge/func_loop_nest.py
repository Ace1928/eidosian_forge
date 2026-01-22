import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
@contextmanager
def loop_nest(builder, shape, intp, order='C'):
    """
    Generate a loop nest walking a N-dimensional array.
    Yields a tuple of N indices for use in the inner loop body,
    iterating over the *shape* space.

    If *order* is 'C' (the default), indices are incremented inside-out
    (i.e. (0,0), (0,1), (0,2), (1,0) etc.).
    If *order* is 'F', they are incremented outside-in
    (i.e. (0,0), (1,0), (2,0), (0,1) etc.).
    This has performance implications when walking an array as it impacts
    the spatial locality of memory accesses.
    """
    assert order in 'CF'
    if not shape:
        yield ()
    else:
        if order == 'F':
            _swap = lambda x: x[::-1]
        else:
            _swap = lambda x: x
        with _loop_nest(builder, _swap(shape), intp) as indices:
            assert len(indices) == len(shape)
            yield _swap(indices)