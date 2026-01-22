from sympy.core import Add, Mul, S
from sympy.core.containers import Tuple
from sympy.core.exprtools import factor_terms
from sympy.core.numbers import I
from sympy.core.relational import Eq, Equality
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Dummy, Symbol
from sympy.core.function import (expand_mul, expand, Derivative,
from sympy.functions import (exp, im, cos, sin, re, Piecewise,
from sympy.functions.combinatorial.factorials import factorial
from sympy.matrices import zeros, Matrix, NonSquareMatrixError, MatrixBase, eye
from sympy.polys import Poly, together
from sympy.simplify import collect, radsimp, signsimp # type: ignore
from sympy.simplify.powsimp import powdenest, powsimp
from sympy.simplify.ratsimp import ratsimp
from sympy.simplify.simplify import simplify
from sympy.sets.sets import FiniteSet
from sympy.solvers.deutils import ode_order
from sympy.solvers.solveset import NonlinearError, solveset
from sympy.utilities.iterables import (connected_components, iterable,
from sympy.utilities.misc import filldedent
from sympy.integrals.integrals import Integral, integrate
def linodesolve_type(A, t, b=None):
    """
    Helper function that determines the type of the system of ODEs for solving with :obj:`sympy.solvers.ode.systems.linodesolve()`

    Explanation
    ===========

    This function takes in the coefficient matrix and/or the non-homogeneous term
    and returns the type of the equation that can be solved by :obj:`sympy.solvers.ode.systems.linodesolve()`.

    If the system is constant coefficient homogeneous, then "type1" is returned

    If the system is constant coefficient non-homogeneous, then "type2" is returned

    If the system is non-constant coefficient homogeneous, then "type3" is returned

    If the system is non-constant coefficient non-homogeneous, then "type4" is returned

    If the system has a non-constant coefficient matrix which can be factorized into constant
    coefficient matrix, then "type5" or "type6" is returned for when the system is homogeneous or
    non-homogeneous respectively.

    Note that, if the system of ODEs is of "type3" or "type4", then along with the type,
    the commutative antiderivative of the coefficient matrix is also returned.

    If the system cannot be solved by :obj:`sympy.solvers.ode.systems.linodesolve()`, then
    NotImplementedError is raised.

    Parameters
    ==========

    A : Matrix
        Coefficient matrix of the system of ODEs
    b : Matrix or None
        Non-homogeneous term of the system. The default value is None.
        If this argument is None, then the system is assumed to be homogeneous.

    Examples
    ========

    >>> from sympy import symbols, Matrix
    >>> from sympy.solvers.ode.systems import linodesolve_type
    >>> t = symbols("t")
    >>> A = Matrix([[1, 1], [2, 3]])
    >>> b = Matrix([t, 1])

    >>> linodesolve_type(A, t)
    {'antiderivative': None, 'type_of_equation': 'type1'}

    >>> linodesolve_type(A, t, b=b)
    {'antiderivative': None, 'type_of_equation': 'type2'}

    >>> A_t = Matrix([[1, t], [-t, 1]])

    >>> linodesolve_type(A_t, t)
    {'antiderivative': Matrix([
    [      t, t**2/2],
    [-t**2/2,      t]]), 'type_of_equation': 'type3'}

    >>> linodesolve_type(A_t, t, b=b)
    {'antiderivative': Matrix([
    [      t, t**2/2],
    [-t**2/2,      t]]), 'type_of_equation': 'type4'}

    >>> A_non_commutative = Matrix([[1, t], [t, -1]])
    >>> linodesolve_type(A_non_commutative, t)
    Traceback (most recent call last):
    ...
    NotImplementedError:
    The system does not have a commutative antiderivative, it cannot be
    solved by linodesolve.

    Returns
    =======

    Dict

    Raises
    ======

    NotImplementedError
        When the coefficient matrix does not have a commutative antiderivative

    See Also
    ========

    linodesolve: Function for which linodesolve_type gets the information

    """
    match = {}
    is_non_constant = not _matrix_is_constant(A, t)
    is_non_homogeneous = not (b is None or b.is_zero_matrix)
    type = 'type{}'.format(int('{}{}'.format(int(is_non_constant), int(is_non_homogeneous)), 2) + 1)
    B = None
    match.update({'type_of_equation': type, 'antiderivative': B})
    if is_non_constant:
        B, is_commuting = _is_commutative_anti_derivative(A, t)
        if not is_commuting:
            raise NotImplementedError(filldedent('\n                The system does not have a commutative antiderivative, it cannot be solved\n                by linodesolve.\n            '))
        match['antiderivative'] = B
        match.update(_first_order_type5_6_subs(A, t, b=b))
    return match