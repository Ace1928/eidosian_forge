from __future__ import annotations
import collections
from functools import reduce
from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.matrices.dense import Matrix
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.core.expr import Expr
from sympy.core.power import Pow
def set_quantity_dimension(self, quantity, dimension):
    """
        Set the dimension for the quantity in a unit system.

        If this relation is valid in every unit system, use
        ``quantity.set_global_dimension(dimension)`` instead.
        """
    from sympy.physics.units import Quantity
    dimension = sympify(dimension)
    if not isinstance(dimension, Dimension):
        if dimension == 1:
            dimension = Dimension(1)
        else:
            raise ValueError('expected dimension or 1')
    elif isinstance(dimension, Quantity):
        dimension = self.get_quantity_dimension(dimension)
    self._quantity_dimension_map[quantity] = dimension