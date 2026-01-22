import itertools
from collections.abc import Iterable
from sympy.core._print_helpers import Printable
from sympy.core.containers import Tuple
from sympy.core.function import diff
from sympy.core.singleton import S
from sympy.core.sympify import _sympify
from sympy.tensor.array.ndim_array import NDimArray
from sympy.tensor.array.dense_ndim_array import DenseNDimArray, ImmutableDenseNDimArray
from sympy.tensor.array.sparse_ndim_array import SparseNDimArray
def derive_by_array(expr, dx):
    """
    Derivative by arrays. Supports both arrays and scalars.

    The equivalent operator for array expressions is ``array_derive``.

    Explanation
    ===========

    Given the array `A_{i_1, \\ldots, i_N}` and the array `X_{j_1, \\ldots, j_M}`
    this function will return a new array `B` defined by

    `B_{j_1,\\ldots,j_M,i_1,\\ldots,i_N} := \\frac{\\partial A_{i_1,\\ldots,i_N}}{\\partial X_{j_1,\\ldots,j_M}}`

    Examples
    ========

    >>> from sympy import derive_by_array
    >>> from sympy.abc import x, y, z, t
    >>> from sympy import cos
    >>> derive_by_array(cos(x*t), x)
    -t*sin(t*x)
    >>> derive_by_array(cos(x*t), [x, y, z, t])
    [-t*sin(t*x), 0, 0, -x*sin(t*x)]
    >>> derive_by_array([x, y**2*z], [[x, y], [z, t]])
    [[[1, 0], [0, 2*y*z]], [[0, y**2], [0, 0]]]

    """
    from sympy.matrices import MatrixBase
    from sympy.tensor.array import SparseNDimArray
    array_types = (Iterable, MatrixBase, NDimArray)
    if isinstance(dx, array_types):
        dx = ImmutableDenseNDimArray(dx)
        for i in dx:
            if not i._diff_wrt:
                raise ValueError('cannot derive by this array')
    if isinstance(expr, array_types):
        if isinstance(expr, NDimArray):
            expr = expr.as_immutable()
        else:
            expr = ImmutableDenseNDimArray(expr)
        if isinstance(dx, array_types):
            if isinstance(expr, SparseNDimArray):
                lp = len(expr)
                new_array = {k + i * lp: v for i, x in enumerate(Flatten(dx)) for k, v in expr.diff(x)._sparse_array.items()}
            else:
                new_array = [[y.diff(x) for y in Flatten(expr)] for x in Flatten(dx)]
            return type(expr)(new_array, dx.shape + expr.shape)
        else:
            return expr.diff(dx)
    else:
        expr = _sympify(expr)
        if isinstance(dx, array_types):
            return ImmutableDenseNDimArray([expr.diff(i) for i in Flatten(dx)], dx.shape)
        else:
            dx = _sympify(dx)
            return diff(expr, dx)