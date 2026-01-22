import cupy
import operator
import numpy
from cupy._core._dtype import get_dtype
def upcast(*args):
    """Returns the nearest supported sparse dtype for the
    combination of one or more types.

    upcast(t0, t1, ..., tn) -> T  where T is a supported dtype

    Examples:
        >>> upcast('int32')
        <type 'numpy.int32'>
        >>> upcast('int32','float32')
        <type 'numpy.float64'>
        >>> upcast('bool',float)
        <type 'numpy.complex128'>
    """
    t = _upcast_memo.get(args)
    if t is not None:
        return t
    upcast = cupy.find_common_type(args, [])
    for t in supported_dtypes:
        if cupy.can_cast(upcast, t):
            _upcast_memo[args] = t
            return t
    raise TypeError('no supported conversion for types: %r' % (args,))