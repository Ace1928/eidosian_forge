from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import connected_components
from sympy.core.sympify import sympify
from sympy.core.numbers import Integer, Rational
from sympy.matrices.dense import MutableDenseMatrix
from sympy.polys.domains import ZZ, QQ
from sympy.polys.domains import EX
from sympy.polys.rings import sring
from sympy.polys.polyerrors import NotInvertible
from sympy.polys.domainmatrix import DomainMatrix
def eqs_to_matrix(eqs_coeffs, eqs_rhs, gens, domain):
    """Get matrix from linear equations in dict format.

    Explanation
    ===========

    Get the matrix representation of a system of linear equations represented
    as dicts with low-level DomainElement coefficients. This is an
    *internal* function that is used by solve_lin_sys.

    Parameters
    ==========

    eqs_coeffs: list[dict[Symbol, DomainElement]]
        The left hand sides of the equations as dicts mapping from symbols to
        coefficients where the coefficients are instances of
        DomainElement.
    eqs_rhs: list[DomainElements]
        The right hand sides of the equations as instances of
        DomainElement.
    gens: list[Symbol]
        The unknowns in the system of equations.
    domain: Domain
        The domain for coefficients of both lhs and rhs.

    Returns
    =======

    The augmented matrix representation of the system as a DomainMatrix.

    Examples
    ========

    >>> from sympy import symbols, ZZ
    >>> from sympy.polys.solvers import eqs_to_matrix
    >>> x, y = symbols('x, y')
    >>> eqs_coeff = [{x:ZZ(1), y:ZZ(1)}, {x:ZZ(1), y:ZZ(-1)}]
    >>> eqs_rhs = [ZZ(0), ZZ(-1)]
    >>> eqs_to_matrix(eqs_coeff, eqs_rhs, [x, y], ZZ)
    DomainMatrix([[1, 1, 0], [1, -1, 1]], (2, 3), ZZ)

    See also
    ========

    solve_lin_sys: Uses :func:`~eqs_to_matrix` internally
    """
    sym2index = {x: n for n, x in enumerate(gens)}
    nrows = len(eqs_coeffs)
    ncols = len(gens) + 1
    rows = [[domain.zero] * ncols for _ in range(nrows)]
    for row, eq_coeff, eq_rhs in zip(rows, eqs_coeffs, eqs_rhs):
        for sym, coeff in eq_coeff.items():
            row[sym2index[sym]] = domain.convert(coeff)
        row[-1] = -domain.convert(eq_rhs)
    return DomainMatrix(rows, (nrows, ncols), domain)