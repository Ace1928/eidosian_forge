import random
from sympy.core.basic import Basic
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from .common import ShapeError
from .decompositions import _cholesky, _LDLdecomposition
from .matrices import MatrixBase
from .repmatrix import MutableRepMatrix, RepMatrix
from .solvers import _lower_triangular_solve, _upper_triangular_solve
def randMatrix(r, c=None, min=0, max=99, seed=None, symmetric=False, percent=100, prng=None):
    """Create random matrix with dimensions ``r`` x ``c``. If ``c`` is omitted
    the matrix will be square. If ``symmetric`` is True the matrix must be
    square. If ``percent`` is less than 100 then only approximately the given
    percentage of elements will be non-zero.

    The pseudo-random number generator used to generate matrix is chosen in the
    following way.

    * If ``prng`` is supplied, it will be used as random number generator.
      It should be an instance of ``random.Random``, or at least have
      ``randint`` and ``shuffle`` methods with same signatures.
    * if ``prng`` is not supplied but ``seed`` is supplied, then new
      ``random.Random`` with given ``seed`` will be created;
    * otherwise, a new ``random.Random`` with default seed will be used.

    Examples
    ========

    >>> from sympy import randMatrix
    >>> randMatrix(3) # doctest:+SKIP
    [25, 45, 27]
    [44, 54,  9]
    [23, 96, 46]
    >>> randMatrix(3, 2) # doctest:+SKIP
    [87, 29]
    [23, 37]
    [90, 26]
    >>> randMatrix(3, 3, 0, 2) # doctest:+SKIP
    [0, 2, 0]
    [2, 0, 1]
    [0, 0, 1]
    >>> randMatrix(3, symmetric=True) # doctest:+SKIP
    [85, 26, 29]
    [26, 71, 43]
    [29, 43, 57]
    >>> A = randMatrix(3, seed=1)
    >>> B = randMatrix(3, seed=2)
    >>> A == B
    False
    >>> A == randMatrix(3, seed=1)
    True
    >>> randMatrix(3, symmetric=True, percent=50) # doctest:+SKIP
    [77, 70,  0],
    [70,  0,  0],
    [ 0,  0, 88]
    """
    prng = prng or random.Random(seed)
    if c is None:
        c = r
    if symmetric and r != c:
        raise ValueError('For symmetric matrices, r must equal c, but %i != %i' % (r, c))
    ij = range(r * c)
    if percent != 100:
        ij = prng.sample(ij, int(len(ij) * percent // 100))
    m = zeros(r, c)
    if not symmetric:
        for ijk in ij:
            i, j = divmod(ijk, c)
            m[i, j] = prng.randint(min, max)
    else:
        for ijk in ij:
            i, j = divmod(ijk, c)
            if i <= j:
                m[i, j] = m[j, i] = prng.randint(min, max)
    return m