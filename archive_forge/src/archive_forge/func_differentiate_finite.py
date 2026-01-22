from sympy.core.function import Derivative
from sympy.core.singleton import S
from sympy.core.function import Subs
from sympy.core.traversal import preorder_traversal
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import iterable
def differentiate_finite(expr, *symbols, points=1, x0=None, wrt=None, evaluate=False):
    """ Differentiate expr and replace Derivatives with finite differences.

    Parameters
    ==========

    expr : expression
    \\*symbols : differentiate with respect to symbols
    points: sequence, coefficient or undefined function, optional
        see ``Derivative.as_finite_difference``
    x0: number or Symbol, optional
        see ``Derivative.as_finite_difference``
    wrt: Symbol, optional
        see ``Derivative.as_finite_difference``

    Examples
    ========

    >>> from sympy import sin, Function, differentiate_finite
    >>> from sympy.abc import x, y, h
    >>> f, g = Function('f'), Function('g')
    >>> differentiate_finite(f(x)*g(x), x, points=[x-h, x+h])
    -f(-h + x)*g(-h + x)/(2*h) + f(h + x)*g(h + x)/(2*h)

    ``differentiate_finite`` works on any expression, including the expressions
    with embedded derivatives:

    >>> differentiate_finite(f(x) + sin(x), x, 2)
    -2*f(x) + f(x - 1) + f(x + 1) - 2*sin(x) + sin(x - 1) + sin(x + 1)
    >>> differentiate_finite(f(x, y), x, y)
    f(x - 1/2, y - 1/2) - f(x - 1/2, y + 1/2) - f(x + 1/2, y - 1/2) + f(x + 1/2, y + 1/2)
    >>> differentiate_finite(f(x)*g(x).diff(x), x)
    (-g(x) + g(x + 1))*f(x + 1/2) - (g(x) - g(x - 1))*f(x - 1/2)

    To make finite difference with non-constant discretization step use
    undefined functions:

    >>> dx = Function('dx')
    >>> differentiate_finite(f(x)*g(x).diff(x), points=dx(x))
    -(-g(x - dx(x)/2 - dx(x - dx(x)/2)/2)/dx(x - dx(x)/2) +
    g(x - dx(x)/2 + dx(x - dx(x)/2)/2)/dx(x - dx(x)/2))*f(x - dx(x)/2)/dx(x) +
    (-g(x + dx(x)/2 - dx(x + dx(x)/2)/2)/dx(x + dx(x)/2) +
    g(x + dx(x)/2 + dx(x + dx(x)/2)/2)/dx(x + dx(x)/2))*f(x + dx(x)/2)/dx(x)

    """
    if any((term.is_Derivative for term in list(preorder_traversal(expr)))):
        evaluate = False
    Dexpr = expr.diff(*symbols, evaluate=evaluate)
    if evaluate:
        sympy_deprecation_warning('\n        The evaluate flag to differentiate_finite() is deprecated.\n\n        evaluate=True expands the intermediate derivatives before computing\n        differences, but this usually not what you want, as it does not\n        satisfy the product rule.\n        ', deprecated_since_version='1.5', active_deprecations_target='deprecated-differentiate_finite-evaluate')
        return Dexpr.replace(lambda arg: arg.is_Derivative, lambda arg: arg.as_finite_difference(points=points, x0=x0, wrt=wrt))
    else:
        DFexpr = Dexpr.as_finite_difference(points=points, x0=x0, wrt=wrt)
        return DFexpr.replace(lambda arg: isinstance(arg, Subs), lambda arg: arg.expr.as_finite_difference(points=points, x0=arg.point[0], wrt=arg.variables[0]))