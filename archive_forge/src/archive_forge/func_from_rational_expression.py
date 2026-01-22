from typing import Type
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.evalf import EvalfMixin
from sympy.core.expr import Expr
from sympy.core.function import expand
from sympy.core.logic import fuzzy_and
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify, _sympify
from sympy.matrices import ImmutableMatrix, eye
from sympy.matrices.expressions import MatMul, MatAdd
from sympy.polys import Poly, rootof
from sympy.polys.polyroots import roots
from sympy.polys.polytools import (cancel, degree)
from sympy.series import limit
from mpmath.libmp.libmpf import prec_to_dps
@classmethod
def from_rational_expression(cls, expr, var=None):
    """
        Creates a new ``TransferFunction`` efficiently from a rational expression.

        Parameters
        ==========

        expr : Expr, Number
            The rational expression representing the ``TransferFunction``.
        var : Symbol, optional
            Complex variable of the Laplace transform used by the
            polynomials of the transfer function.

        Raises
        ======

        ValueError
            When ``expr`` is of type ``Number`` and optional parameter ``var``
            is not passed.

            When ``expr`` has more than one variables and an optional parameter
            ``var`` is not passed.
        ZeroDivisionError
            When denominator of ``expr`` is zero or it has ``ComplexInfinity``
            in its numerator.

        Examples
        ========

        >>> from sympy.abc import s, p, a
        >>> from sympy.physics.control.lti import TransferFunction
        >>> expr1 = (s + 5)/(3*s**2 + 2*s + 1)
        >>> tf1 = TransferFunction.from_rational_expression(expr1)
        >>> tf1
        TransferFunction(s + 5, 3*s**2 + 2*s + 1, s)
        >>> expr2 = (a*p**3 - a*p**2 + s*p)/(p + a**2)  # Expr with more than one variables
        >>> tf2 = TransferFunction.from_rational_expression(expr2, p)
        >>> tf2
        TransferFunction(a*p**3 - a*p**2 + p*s, a**2 + p, p)

        In case of conflict between two or more variables in a expression, SymPy will
        raise a ``ValueError``, if ``var`` is not passed by the user.

        >>> tf = TransferFunction.from_rational_expression((a + a*s)/(s**2 + s + 1))
        Traceback (most recent call last):
        ...
        ValueError: Conflicting values found for positional argument `var` ({a, s}). Specify it manually.

        This can be corrected by specifying the ``var`` parameter manually.

        >>> tf = TransferFunction.from_rational_expression((a + a*s)/(s**2 + s + 1), s)
        >>> tf
        TransferFunction(a*s + a, s**2 + s + 1, s)

        ``var`` also need to be specified when ``expr`` is a ``Number``

        >>> tf3 = TransferFunction.from_rational_expression(10, s)
        >>> tf3
        TransferFunction(10, 1, s)

        """
    expr = _sympify(expr)
    if var is None:
        _free_symbols = expr.free_symbols
        _len_free_symbols = len(_free_symbols)
        if _len_free_symbols == 1:
            var = list(_free_symbols)[0]
        elif _len_free_symbols == 0:
            raise ValueError('Positional argument `var` not found in the TransferFunction defined. Specify it manually.')
        else:
            raise ValueError('Conflicting values found for positional argument `var` ({}). Specify it manually.'.format(_free_symbols))
    _num, _den = expr.as_numer_denom()
    if _den == 0 or _num.has(S.ComplexInfinity):
        raise ZeroDivisionError('TransferFunction cannot have a zero denominator.')
    return cls(_num, _den, var)