from sympy.logic.boolalg import And, Not, conjuncts, to_cnf, BooleanFunction
from sympy.core.sorting import ordered
from sympy.core.sympify import sympify
from sympy.external.importtools import import_module
def literal_symbol(literal):
    """
    The symbol in this literal (without the negation).

    Examples
    ========

    >>> from sympy.abc import A
    >>> from sympy.logic.inference import literal_symbol
    >>> literal_symbol(A)
    A
    >>> literal_symbol(~A)
    A

    """
    if literal is True or literal is False:
        return literal
    try:
        if literal.is_Symbol:
            return literal
        if literal.is_Not:
            return literal_symbol(literal.args[0])
        else:
            raise ValueError
    except (AttributeError, ValueError):
        raise ValueError('Argument must be a boolean literal.')