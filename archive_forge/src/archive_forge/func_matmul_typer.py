import warnings
import numpy as np
import operator
from numba.core import types, utils, config
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.np.numpy_support import (ufunc_find_matching_loop,
from numba.core.errors import (TypingError, NumbaPerformanceWarning,
from numba import pndindex
def matmul_typer(self, a, b, out=None):
    """
        Typer function for Numpy matrix multiplication.
        """
    if not isinstance(a, types.Array) or not isinstance(b, types.Array):
        return
    if not all((x.ndim in (1, 2) for x in (a, b))):
        raise TypingError('%s only supported on 1-D and 2-D arrays' % (self.func_name,))
    ndims = set([a.ndim, b.ndim])
    if ndims == set([2]):
        out_ndim = 2
    elif ndims == set([1, 2]):
        out_ndim = 1
    elif ndims == set([1]):
        out_ndim = 0
    if out is not None:
        if out_ndim == 0:
            raise TypeError('explicit output unsupported for vector * vector')
        elif out.ndim != out_ndim:
            raise TypeError('explicit output has incorrect dimensionality')
        if not isinstance(out, types.Array) or out.layout != 'C':
            raise TypeError('output must be a C-contiguous array')
        all_args = (a, b, out)
    else:
        all_args = (a, b)
    if not (config.DISABLE_PERFORMANCE_WARNINGS or all((x.layout in 'CF' for x in (a, b)))):
        msg = '%s is faster on contiguous arrays, called on %s' % (self.func_name, (a, b))
        warnings.warn(NumbaPerformanceWarning(msg))
    if not all((x.dtype == a.dtype for x in all_args)):
        raise TypingError('%s arguments must all have the same dtype' % (self.func_name,))
    if not isinstance(a.dtype, (types.Float, types.Complex)):
        raise TypingError('%s only supported on float and complex arrays' % (self.func_name,))
    if out:
        return out
    elif out_ndim > 0:
        return types.Array(a.dtype, out_ndim, 'C')
    else:
        return a.dtype