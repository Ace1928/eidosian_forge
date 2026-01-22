from functools import reduce
from collections.abc import Iterable
from typing import Optional
from sympy import default_sort_key
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.sorting import ordered
from sympy.core.sympify import sympify
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.physics.units.dimensions import Dimension, DimensionSystem
from sympy.physics.units.prefixes import Prefix
from sympy.physics.units.quantities import Quantity
from sympy.physics.units.unitsystem import UnitSystem
from sympy.utilities.iterables import sift
def get_total_scale_factor(expr):
    if isinstance(expr, Mul):
        return reduce(lambda x, y: x * y, [get_total_scale_factor(i) for i in expr.args])
    elif isinstance(expr, Pow):
        return get_total_scale_factor(expr.base) ** expr.exp
    elif isinstance(expr, Quantity):
        return unit_system.get_quantity_scale_factor(expr)
    return expr