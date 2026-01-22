from typing import Dict as tDict, Set as tSet
from sympy.core.add import Add
from sympy.core.function import (Derivative, Function)
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.physics.units.dimensions import _QuantityMapper
from sympy.physics.units.quantities import Quantity
from .dimensions import Dimension
def get_dimensional_expr(self, expr):
    from sympy.physics.units import Quantity
    if isinstance(expr, Mul):
        return Mul(*[self.get_dimensional_expr(i) for i in expr.args])
    elif isinstance(expr, Pow):
        return self.get_dimensional_expr(expr.base) ** expr.exp
    elif isinstance(expr, Add):
        return self.get_dimensional_expr(expr.args[0])
    elif isinstance(expr, Derivative):
        dim = self.get_dimensional_expr(expr.expr)
        for independent, count in expr.variable_count:
            dim /= self.get_dimensional_expr(independent) ** count
        return dim
    elif isinstance(expr, Function):
        args = [self.get_dimensional_expr(arg) for arg in expr.args]
        if all((i == 1 for i in args)):
            return S.One
        return expr.func(*args)
    elif isinstance(expr, Quantity):
        return self.get_quantity_dimension(expr).name
    return S.One