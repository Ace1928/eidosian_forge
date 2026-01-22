import warnings
from sympy.core.add import Add
from sympy.core.function import (Function, diff)
from sympy.core.numbers import (Number, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.integrals.integrals import integrate
from sympy.physics.units import (amount_of_substance, area, convert_to, find_unit,
from sympy.physics.units.definitions import (amu, au, centimeter, coulomb,
from sympy.physics.units.definitions.dimension_definitions import (
from sympy.physics.units.prefixes import PREFIXES, kilo
from sympy.physics.units.quantities import PhysicalConstant, Quantity
from sympy.physics.units.systems import SI
from sympy.testing.pytest import raises
def test_physics_constant():
    from sympy.physics.units import definitions
    for name in dir(definitions):
        quantity = getattr(definitions, name)
        if not isinstance(quantity, Quantity):
            continue
        if name.endswith('_constant'):
            assert isinstance(quantity, PhysicalConstant), f'{quantity} must be PhysicalConstant, but is {type(quantity)}'
            assert quantity.is_physical_constant, f'{name} is not marked as physics constant when it should be'
    for const in [gravitational_constant, molar_gas_constant, vacuum_permittivity, speed_of_light, elementary_charge]:
        assert isinstance(const, PhysicalConstant), f'{const} must be PhysicalConstant, but is {type(const)}'
        assert const.is_physical_constant, f'{const} is not marked as physics constant when it should be'
    assert not meter.is_physical_constant
    assert not joule.is_physical_constant