from sympy.utilities.exceptions import sympy_deprecation_warning
from .facts import FactRules, FactKB
from .sympify import sympify
from sympy.core.random import _assumptions_shuffle as shuffle
from sympy.core.assumptions_generated import generated_assumptions as _assumptions
def failing_assumptions(expr, **assumptions):
    """
    Return a dictionary containing assumptions with values not
    matching those of the passed assumptions.

    Examples
    ========

    >>> from sympy import failing_assumptions, Symbol

    >>> x = Symbol('x', positive=True)
    >>> y = Symbol('y')
    >>> failing_assumptions(6*x + y, positive=True)
    {'positive': None}

    >>> failing_assumptions(x**2 - 1, positive=True)
    {'positive': None}

    If *expr* satisfies all of the assumptions, an empty dictionary is returned.

    >>> failing_assumptions(x**2, positive=True)
    {}

    """
    expr = sympify(expr)
    failed = {}
    for k in assumptions:
        test = getattr(expr, 'is_%s' % k, None)
        if test is not assumptions[k]:
            failed[k] = test
    return failed