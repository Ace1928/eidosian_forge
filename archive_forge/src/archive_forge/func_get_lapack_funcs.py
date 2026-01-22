import numpy as _np
from .blas import _get_funcs, _memoize_get_funcs
from scipy.linalg import _flapack
from re import compile as regex_compile
from scipy.linalg._flapack import *  # noqa: E402, F403
@_memoize_get_funcs
def get_lapack_funcs(names, arrays=(), dtype=None, ilp64=False):
    """Return available LAPACK function objects from names.

    Arrays are used to determine the optimal prefix of LAPACK routines.

    Parameters
    ----------
    names : str or sequence of str
        Name(s) of LAPACK functions without type prefix.

    arrays : sequence of ndarrays, optional
        Arrays can be given to determine optimal prefix of LAPACK
        routines. If not given, double-precision routines will be
        used, otherwise the most generic type in arrays will be used.

    dtype : str or dtype, optional
        Data-type specifier. Not used if `arrays` is non-empty.

    ilp64 : {True, False, 'preferred'}, optional
        Whether to return ILP64 routine variant.
        Choosing 'preferred' returns ILP64 routine if available, and
        otherwise the 32-bit routine. Default: False

    Returns
    -------
    funcs : list
        List containing the found function(s).

    Notes
    -----
    This routine automatically chooses between Fortran/C
    interfaces. Fortran code is used whenever possible for arrays with
    column major order. In all other cases, C code is preferred.

    In LAPACK, the naming convention is that all functions start with a
    type prefix, which depends on the type of the principal
    matrix. These can be one of {'s', 'd', 'c', 'z'} for the NumPy
    types {float32, float64, complex64, complex128} respectively, and
    are stored in attribute ``typecode`` of the returned functions.

    Examples
    --------
    Suppose we would like to use '?lange' routine which computes the selected
    norm of an array. We pass our array in order to get the correct 'lange'
    flavor.

    >>> import numpy as np
    >>> import scipy.linalg as LA
    >>> rng = np.random.default_rng()

    >>> a = rng.random((3,2))
    >>> x_lange = LA.get_lapack_funcs('lange', (a,))
    >>> x_lange.typecode
    'd'
    >>> x_lange = LA.get_lapack_funcs('lange',(a*1j,))
    >>> x_lange.typecode
    'z'

    Several LAPACK routines work best when its internal WORK array has
    the optimal size (big enough for fast computation and small enough to
    avoid waste of memory). This size is determined also by a dedicated query
    to the function which is often wrapped as a standalone function and
    commonly denoted as ``###_lwork``. Below is an example for ``?sysv``

    >>> a = rng.random((1000, 1000))
    >>> b = rng.random((1000, 1)) * 1j
    >>> # We pick up zsysv and zsysv_lwork due to b array
    ... xsysv, xlwork = LA.get_lapack_funcs(('sysv', 'sysv_lwork'), (a, b))
    >>> opt_lwork, _ = xlwork(a.shape[0])  # returns a complex for 'z' prefix
    >>> udut, ipiv, x, info = xsysv(a, b, lwork=int(opt_lwork.real))

    """
    if isinstance(ilp64, str):
        if ilp64 == 'preferred':
            ilp64 = HAS_ILP64
        else:
            raise ValueError("Invalid value for 'ilp64'")
    if not ilp64:
        return _get_funcs(names, arrays, dtype, 'LAPACK', _flapack, _clapack, 'flapack', 'clapack', _lapack_alias, ilp64=False)
    else:
        if not HAS_ILP64:
            raise RuntimeError('LAPACK ILP64 routine requested, but Scipy compiled only with 32-bit BLAS')
        return _get_funcs(names, arrays, dtype, 'LAPACK', _flapack_64, None, 'flapack_64', None, _lapack_alias, ilp64=True)