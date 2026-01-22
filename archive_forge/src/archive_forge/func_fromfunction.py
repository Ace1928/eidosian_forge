import numpy
from cupy import _core
from cupy._core import fusion
def fromfunction(*args, **kwargs):
    """Construct an array by executing a function over each coordinate.

    .. note::
        Uses NumPy's ``fromfunction`` and coerces the result to a CuPy array.

    .. seealso:: :func:`numpy.fromfunction`
    """
    return asarray(numpy.fromfunction(*args, **kwargs))