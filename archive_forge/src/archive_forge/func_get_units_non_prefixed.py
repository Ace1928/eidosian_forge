from typing import Dict as tDict, Set as tSet
from sympy.core.add import Add
from sympy.core.function import (Derivative, Function)
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.physics.units.dimensions import _QuantityMapper
from sympy.physics.units.quantities import Quantity
from .dimensions import Dimension
def get_units_non_prefixed(self) -> tSet[Quantity]:
    """
        Return the units of the system that do not have a prefix.
        """
    return set(filter(lambda u: not u.is_prefixed and (not u.is_physical_constant), self._units))