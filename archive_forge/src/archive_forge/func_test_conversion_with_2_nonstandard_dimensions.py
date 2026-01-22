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
def test_conversion_with_2_nonstandard_dimensions():
    good_grade = Quantity('good_grade')
    kilo_good_grade = Quantity('kilo_good_grade')
    centi_good_grade = Quantity('centi_good_grade')
    kilo_good_grade.set_global_relative_scale_factor(1000, good_grade)
    centi_good_grade.set_global_relative_scale_factor(S.One / 10 ** 5, kilo_good_grade)
    charity_points = Quantity('charity_points')
    milli_charity_points = Quantity('milli_charity_points')
    missions = Quantity('missions')
    milli_charity_points.set_global_relative_scale_factor(S.One / 1000, charity_points)
    missions.set_global_relative_scale_factor(251, charity_points)
    assert convert_to(kilo_good_grade * milli_charity_points * millimeter, [centi_good_grade, missions, centimeter]) == S.One * 10 ** 5 / (251 * 1000) / 10 * centi_good_grade * missions * centimeter