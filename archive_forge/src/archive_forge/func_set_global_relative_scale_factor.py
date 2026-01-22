from sympy.core.expr import AtomicExpr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.physics.units.dimensions import _QuantityMapper
from sympy.physics.units.prefixes import Prefix
def set_global_relative_scale_factor(self, scale_factor, reference_quantity):
    """
        Setting a scale factor that is valid across all unit system.
        """
    from sympy.physics.units import UnitSystem
    scale_factor = sympify(scale_factor)
    if isinstance(scale_factor, Prefix):
        self._is_prefixed = True
    scale_factor = scale_factor.replace(lambda x: isinstance(x, Prefix), lambda x: x.scale_factor)
    scale_factor = sympify(scale_factor)
    UnitSystem._quantity_scale_factors_global[self] = (scale_factor, reference_quantity)
    UnitSystem._quantity_dimensional_equivalence_map_global[self] = reference_quantity