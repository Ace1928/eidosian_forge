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
def get_dimensional_dependencies(self, name, mark_dimensionless=False):
    dimdep = self._get_dimensional_dependencies_for_name(name)
    if mark_dimensionless and dimdep == {}:
        return {Dimension(1): 1}
    return {k: v for k, v in dimdep.items()}