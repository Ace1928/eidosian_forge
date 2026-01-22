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
def test_find_unit():
    assert find_unit('coulomb') == ['coulomb', 'coulombs', 'coulomb_constant']
    assert find_unit(coulomb) == ['C', 'coulomb', 'coulombs', 'planck_charge', 'elementary_charge']
    assert find_unit(charge) == ['C', 'coulomb', 'coulombs', 'planck_charge', 'elementary_charge']
    assert find_unit(inch) == ['m', 'au', 'cm', 'dm', 'ft', 'km', 'ly', 'mi', 'mm', 'nm', 'pm', 'um', 'yd', 'nmi', 'feet', 'foot', 'inch', 'mile', 'yard', 'meter', 'miles', 'yards', 'inches', 'meters', 'micron', 'microns', 'angstrom', 'angstroms', 'decimeter', 'kilometer', 'lightyear', 'nanometer', 'picometer', 'centimeter', 'decimeters', 'kilometers', 'lightyears', 'micrometer', 'millimeter', 'nanometers', 'picometers', 'centimeters', 'micrometers', 'millimeters', 'nautical_mile', 'planck_length', 'nautical_miles', 'astronomical_unit', 'astronomical_units']
    assert find_unit(inch ** (-1)) == ['D', 'dioptre', 'optical_power']
    assert find_unit(length ** (-1)) == ['D', 'dioptre', 'optical_power']
    assert find_unit(inch ** 2) == ['ha', 'hectare', 'planck_area']
    assert find_unit(inch ** 3) == ['L', 'l', 'cL', 'cl', 'dL', 'dl', 'mL', 'ml', 'liter', 'quart', 'liters', 'quarts', 'deciliter', 'centiliter', 'deciliters', 'milliliter', 'centiliters', 'milliliters', 'planck_volume']
    assert find_unit('voltage') == ['V', 'v', 'volt', 'volts', 'planck_voltage']
    assert find_unit(grams) == ['g', 't', 'Da', 'kg', 'me', 'mg', 'ug', 'amu', 'mmu', 'amus', 'gram', 'mmus', 'grams', 'pound', 'tonne', 'dalton', 'pounds', 'kilogram', 'kilograms', 'microgram', 'milligram', 'metric_ton', 'micrograms', 'milligrams', 'planck_mass', 'milli_mass_unit', 'atomic_mass_unit', 'electron_rest_mass', 'atomic_mass_constant']