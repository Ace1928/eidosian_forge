import sympy
from sympy import S
from sympy.core.logic import fuzzy_and, fuzzy_not, fuzzy_or
def check_supported_type(x):
    if x.is_integer is False and x.is_real is False and x.is_complex or x.is_Boolean:
        raise TypeError(f"unsupported operand type(s) for //: '{type(base).__name__}' and '{type(divisor).__name__}', expected integer or real")