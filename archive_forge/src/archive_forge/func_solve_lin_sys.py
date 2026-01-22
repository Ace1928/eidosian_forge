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
def solve_lin_sys(eqs, ring, _raw=True):
    """Solve a system of linear equations from a PolynomialRing

    Explanation
    ===========

    Solves a system of linear equations given as PolyElement instances of a
    PolynomialRing. The basic arithmetic is carried out using instance of
    DomainElement which is more efficient than :class:`~sympy.core.expr.Expr`
    for the most common inputs.

    While this is a public function it is intended primarily for internal use
    so its interface is not necessarily convenient. Users are suggested to use
    the :func:`sympy.solvers.solveset.linsolve` function (which uses this
    function internally) instead.

    Parameters
    ==========

    eqs: list[PolyElement]
        The linear equations to be solved as elements of a
        PolynomialRing (assumed equal to zero).
    ring: PolynomialRing
        The polynomial ring from which eqs are drawn. The generators of this
        ring are the unknowns to be solved for and the domain of the ring is
        the domain of the coefficients of the system of equations.
    _raw: bool
        If *_raw* is False, the keys and values in the returned dictionary
        will be of type Expr (and the unit of the field will be removed from
        the keys) otherwise the low-level polys types will be returned, e.g.
        PolyElement: PythonRational.

    Returns
    =======

    ``None`` if the system has no solution.

    dict[Symbol, Expr] if _raw=False

    dict[Symbol, DomainElement] if _raw=True.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.polys.solvers import solve_lin_sys, sympy_eqs_to_ring
    >>> x, y = symbols('x, y')
    >>> eqs = [x - y, x + y - 2]
    >>> eqs_ring, ring = sympy_eqs_to_ring(eqs, [x, y])
    >>> solve_lin_sys(eqs_ring, ring)
    {y: 1, x: 1}

    Passing ``_raw=False`` returns the same result except that the keys are
    ``Expr`` rather than low-level poly types.

    >>> solve_lin_sys(eqs_ring, ring, _raw=False)
    {x: 1, y: 1}

    See also
    ========

    sympy_eqs_to_ring: prepares the inputs to ``solve_lin_sys``.
    linsolve: ``linsolve`` uses ``solve_lin_sys`` internally.
    sympy.solvers.solvers.solve: ``solve`` uses ``solve_lin_sys`` internally.
    """
    as_expr = not _raw
    assert ring.domain.is_Field
    eqs_dict = [dict(eq) for eq in eqs]
    one_monom = ring.one.monoms()[0]
    zero = ring.domain.zero
    eqs_rhs = []
    eqs_coeffs = []
    for eq_dict in eqs_dict:
        eq_rhs = eq_dict.pop(one_monom, zero)
        eq_coeffs = {}
        for monom, coeff in eq_dict.items():
            if sum(monom) != 1:
                msg = 'Nonlinear term encountered in solve_lin_sys'
                raise PolyNonlinearError(msg)
            eq_coeffs[ring.gens[monom.index(1)]] = coeff
        if not eq_coeffs:
            if not eq_rhs:
                continue
            else:
                return None
        eqs_rhs.append(eq_rhs)
        eqs_coeffs.append(eq_coeffs)
    result = _solve_lin_sys(eqs_coeffs, eqs_rhs, ring)
    if result is not None and as_expr:

        def to_sympy(x):
            as_expr = getattr(x, 'as_expr', None)
            if as_expr:
                return as_expr()
            else:
                return ring.domain.to_sympy(x)
        tresult = {to_sympy(sym): to_sympy(val) for sym, val in result.items()}
        result = {}
        for k, v in tresult.items():
            if k.is_Mul:
                c, s = k.as_coeff_Mul()
                result[s] = v / c
            else:
                result[k] = v
    return result