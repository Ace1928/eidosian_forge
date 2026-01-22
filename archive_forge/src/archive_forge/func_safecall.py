import numpy
from .lapack import get_lapack_funcs
from ._misc import _datacopied
def safecall(f, name, *args, **kwargs):
    """Call a LAPACK routine, determining lwork automatically and handling
    error return values"""
    lwork = kwargs.get('lwork', None)
    if lwork in (None, -1):
        kwargs['lwork'] = -1
        ret = f(*args, **kwargs)
        kwargs['lwork'] = ret[-2][0].real.astype(numpy.int_)
    ret = f(*args, **kwargs)
    if ret[-1] < 0:
        raise ValueError('illegal value in %dth argument of internal %s' % (-ret[-1], name))
    return ret[:-2]