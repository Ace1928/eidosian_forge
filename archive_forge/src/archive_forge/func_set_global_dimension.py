from sympy.core.expr import AtomicExpr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.physics.units.dimensions import _QuantityMapper
from sympy.physics.units.prefixes import Prefix
def set_global_dimension(self, dimension):
    _QuantityMapper._quantity_dimension_global[self] = dimension