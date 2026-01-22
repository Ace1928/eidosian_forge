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
def set_quantity_scale_factor(self, quantity, scale_factor):
    """
        Set the scale factor of a quantity relative to another quantity.

        It should be used only once per quantity to just one other quantity,
        the algorithm will then be able to compute the scale factors to all
        other quantities.

        In case the scale factor is valid in every unit system, please use
        ``quantity.set_global_relative_scale_factor(scale_factor)`` instead.
        """
    from sympy.physics.units import Quantity
    from sympy.physics.units.prefixes import Prefix
    scale_factor = sympify(scale_factor)
    scale_factor = scale_factor.replace(lambda x: isinstance(x, Prefix), lambda x: x.scale_factor)
    scale_factor = scale_factor.replace(lambda x: isinstance(x, Quantity), lambda x: self.get_quantity_scale_factor(x))
    self._quantity_scale_factors[quantity] = scale_factor