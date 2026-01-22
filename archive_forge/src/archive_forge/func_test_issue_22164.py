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
def test_issue_22164():
    warnings.simplefilter('error')
    dm = Quantity('dm')
    SI.set_quantity_dimension(dm, length)
    SI.set_quantity_scale_factor(dm, 1)
    bad_exp = Quantity('bad_exp')
    SI.set_quantity_dimension(bad_exp, length)
    SI.set_quantity_scale_factor(bad_exp, 1)
    expr = dm ** bad_exp
    SI._collect_factor_and_dimension(expr)