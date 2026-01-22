from sympy.physics.units import DimensionSystem, joule, second, ampere
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.physics.units.definitions import c, kg, m, s
from sympy.physics.units.definitions.dimension_definitions import length, time
from sympy.physics.units.quantities import Quantity
from sympy.physics.units.unitsystem import UnitSystem
from sympy.physics.units.util import convert_to
def test_derived_units_must_exist_in_unit_system():
    for unit_system in UnitSystem._unit_systems.values():
        for preferred_unit in unit_system.derived_units.values():
            units = preferred_unit.atoms(Quantity)
            for unit in units:
                assert unit in unit_system._units, f'Unit {unit} is not in unit system {unit_system}'