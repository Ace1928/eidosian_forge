from sympy.assumptions.ask import Q
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.symbol import symbols
from sympy.logic.boolalg import (And, Or)
from sympy.assumptions.sathandlers import (ClassFactRegistry, allargs,
@my_handler_registry.register(Mul)
def fact1(expr):
    pass