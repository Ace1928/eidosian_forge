from sympy.core.function import Derivative
from sympy.core.singleton import S
from sympy.core.function import Subs
from sympy.core.traversal import preorder_traversal
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import iterable
def finite_diff_weights(order, x_list, x0=S.One):
    """
    Calculates the finite difference weights for an arbitrarily spaced
    one-dimensional grid (``x_list``) for derivatives at ``x0`` of order
    0, 1, ..., up to ``order`` using a recursive formula. Order of accuracy
    is at least ``len(x_list) - order``, if ``x_list`` is defined correctly.

    Parameters
    ==========

    order: int
        Up to what derivative order weights should be calculated.
        0 corresponds to interpolation.
    x_list: sequence
        Sequence of (unique) values for the independent variable.
        It is useful (but not necessary) to order ``x_list`` from
        nearest to furthest from ``x0``; see examples below.
    x0: Number or Symbol
        Root or value of the independent variable for which the finite
        difference weights should be generated. Default is ``S.One``.

    Returns
    =======

    list
        A list of sublists, each corresponding to coefficients for
        increasing derivative order, and each containing lists of
        coefficients for increasing subsets of x_list.

    Examples
    ========

    >>> from sympy import finite_diff_weights, S
    >>> res = finite_diff_weights(1, [-S(1)/2, S(1)/2, S(3)/2, S(5)/2], 0)
    >>> res
    [[[1, 0, 0, 0],
      [1/2, 1/2, 0, 0],
      [3/8, 3/4, -1/8, 0],
      [5/16, 15/16, -5/16, 1/16]],
     [[0, 0, 0, 0],
      [-1, 1, 0, 0],
      [-1, 1, 0, 0],
      [-23/24, 7/8, 1/8, -1/24]]]
    >>> res[0][-1]  # FD weights for 0th derivative, using full x_list
    [5/16, 15/16, -5/16, 1/16]
    >>> res[1][-1]  # FD weights for 1st derivative
    [-23/24, 7/8, 1/8, -1/24]
    >>> res[1][-2]  # FD weights for 1st derivative, using x_list[:-1]
    [-1, 1, 0, 0]
    >>> res[1][-1][0]  # FD weight for 1st deriv. for x_list[0]
    -23/24
    >>> res[1][-1][1]  # FD weight for 1st deriv. for x_list[1], etc.
    7/8

    Each sublist contains the most accurate formula at the end.
    Note, that in the above example ``res[1][1]`` is the same as ``res[1][2]``.
    Since res[1][2] has an order of accuracy of
    ``len(x_list[:3]) - order = 3 - 1 = 2``, the same is true for ``res[1][1]``!

    >>> res = finite_diff_weights(1, [S(0), S(1), -S(1), S(2), -S(2)], 0)[1]
    >>> res
    [[0, 0, 0, 0, 0],
     [-1, 1, 0, 0, 0],
     [0, 1/2, -1/2, 0, 0],
     [-1/2, 1, -1/3, -1/6, 0],
     [0, 2/3, -2/3, -1/12, 1/12]]
    >>> res[0]  # no approximation possible, using x_list[0] only
    [0, 0, 0, 0, 0]
    >>> res[1]  # classic forward step approximation
    [-1, 1, 0, 0, 0]
    >>> res[2]  # classic centered approximation
    [0, 1/2, -1/2, 0, 0]
    >>> res[3:]  # higher order approximations
    [[-1/2, 1, -1/3, -1/6, 0], [0, 2/3, -2/3, -1/12, 1/12]]

    Let us compare this to a differently defined ``x_list``. Pay attention to
    ``foo[i][k]`` corresponding to the gridpoint defined by ``x_list[k]``.

    >>> foo = finite_diff_weights(1, [-S(2), -S(1), S(0), S(1), S(2)], 0)[1]
    >>> foo
    [[0, 0, 0, 0, 0],
     [-1, 1, 0, 0, 0],
     [1/2, -2, 3/2, 0, 0],
     [1/6, -1, 1/2, 1/3, 0],
     [1/12, -2/3, 0, 2/3, -1/12]]
    >>> foo[1]  # not the same and of lower accuracy as res[1]!
    [-1, 1, 0, 0, 0]
    >>> foo[2]  # classic double backward step approximation
    [1/2, -2, 3/2, 0, 0]
    >>> foo[4]  # the same as res[4]
    [1/12, -2/3, 0, 2/3, -1/12]

    Note that, unless you plan on using approximations based on subsets of
    ``x_list``, the order of gridpoints does not matter.

    The capability to generate weights at arbitrary points can be
    used e.g. to minimize Runge's phenomenon by using Chebyshev nodes:

    >>> from sympy import cos, symbols, pi, simplify
    >>> N, (h, x) = 4, symbols('h x')
    >>> x_list = [x+h*cos(i*pi/(N)) for i in range(N,-1,-1)] # chebyshev nodes
    >>> print(x_list)
    [-h + x, -sqrt(2)*h/2 + x, x, sqrt(2)*h/2 + x, h + x]
    >>> mycoeffs = finite_diff_weights(1, x_list, 0)[1][4]
    >>> [simplify(c) for c in  mycoeffs] #doctest: +NORMALIZE_WHITESPACE
    [(h**3/2 + h**2*x - 3*h*x**2 - 4*x**3)/h**4,
    (-sqrt(2)*h**3 - 4*h**2*x + 3*sqrt(2)*h*x**2 + 8*x**3)/h**4,
    (6*h**2*x - 8*x**3)/h**4,
    (sqrt(2)*h**3 - 4*h**2*x - 3*sqrt(2)*h*x**2 + 8*x**3)/h**4,
    (-h**3/2 + h**2*x + 3*h*x**2 - 4*x**3)/h**4]

    Notes
    =====

    If weights for a finite difference approximation of 3rd order
    derivative is wanted, weights for 0th, 1st and 2nd order are
    calculated "for free", so are formulae using subsets of ``x_list``.
    This is something one can take advantage of to save computational cost.
    Be aware that one should define ``x_list`` from nearest to furthest from
    ``x0``. If not, subsets of ``x_list`` will yield poorer approximations,
    which might not grand an order of accuracy of ``len(x_list) - order``.

    See also
    ========

    sympy.calculus.finite_diff.apply_finite_diff

    References
    ==========

    .. [1] Generation of Finite Difference Formulas on Arbitrarily Spaced
            Grids, Bengt Fornberg; Mathematics of computation; 51; 184;
            (1988); 699-706; doi:10.1090/S0025-5718-1988-0935077-0

    """
    order = S(order)
    if not order.is_number:
        raise ValueError('Cannot handle symbolic order.')
    if order < 0:
        raise ValueError('Negative derivative order illegal.')
    if int(order) != order:
        raise ValueError('Non-integer order illegal')
    M = order
    N = len(x_list) - 1
    delta = [[[0 for nu in range(N + 1)] for n in range(N + 1)] for m in range(M + 1)]
    delta[0][0][0] = S.One
    c1 = S.One
    for n in range(1, N + 1):
        c2 = S.One
        for nu in range(n):
            c3 = x_list[n] - x_list[nu]
            c2 = c2 * c3
            if n <= M:
                delta[n][n - 1][nu] = 0
            for m in range(min(n, M) + 1):
                delta[m][n][nu] = (x_list[n] - x0) * delta[m][n - 1][nu] - m * delta[m - 1][n - 1][nu]
                delta[m][n][nu] /= c3
        for m in range(min(n, M) + 1):
            delta[m][n][n] = c1 / c2 * (m * delta[m - 1][n - 1][n - 1] - (x_list[n - 1] - x0) * delta[m][n - 1][n - 1])
        c1 = c2
    return delta