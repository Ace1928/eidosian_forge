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
def get_quantity_scale_factor(self, unit):
    if unit in self._quantity_scale_factors:
        return self._quantity_scale_factors[unit]
    if unit in self._quantity_scale_factors_global:
        mul_factor, other_unit = self._quantity_scale_factors_global[unit]
        return mul_factor * self.get_quantity_scale_factor(other_unit)
    return S.One