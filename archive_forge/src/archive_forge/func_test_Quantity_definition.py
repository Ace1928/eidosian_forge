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
def test_Quantity_definition():
    q = Quantity('s10', abbrev='sabbr')
    q.set_global_relative_scale_factor(10, second)
    u = Quantity('u', abbrev='dam')
    u.set_global_relative_scale_factor(10, meter)
    km = Quantity('km')
    km.set_global_relative_scale_factor(kilo, meter)
    v = Quantity('u')
    v.set_global_relative_scale_factor(5 * kilo, meter)
    assert q.scale_factor == 10
    assert q.dimension == time
    assert q.abbrev == Symbol('sabbr')
    assert u.dimension == length
    assert u.scale_factor == 10
    assert u.abbrev == Symbol('dam')
    assert km.scale_factor == 1000
    assert km.func(*km.args) == km
    assert km.func(*km.args).args == km.args
    assert v.dimension == length
    assert v.scale_factor == 5000