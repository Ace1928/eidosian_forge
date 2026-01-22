from typing import Dict as tDict, Set as tSet
from sympy.core.add import Add
from sympy.core.function import (Derivative, Function)
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.physics.units.dimensions import _QuantityMapper
from sympy.physics.units.quantities import Quantity
from .dimensions import Dimension
@staticmethod
def get_default_unit_system():
    return UnitSystem._unit_systems['SI']