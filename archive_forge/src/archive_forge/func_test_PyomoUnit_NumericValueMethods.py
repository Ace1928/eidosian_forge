import math
import pickle
from pyomo.common.errors import PyomoException
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.log import LoggingIntercept
from pyomo.util.check_units import assert_units_consistent, check_units_equivalent
from pyomo.core.expr import inequality
from pyomo.core.expr.numvalue import NumericConstant
import pyomo.core.expr as EXPR
from pyomo.core.base.units_container import (
from io import StringIO
def test_PyomoUnit_NumericValueMethods(self):
    m = ConcreteModel()
    uc = units
    kg = uc.kg
    self.assertEqual(kg.getname(), 'kg')
    self.assertEqual(kg.name, 'kg')
    self.assertEqual(kg.local_name, 'kg')
    m.kg = uc.kg
    self.assertEqual(m.kg.name, 'kg')
    self.assertEqual(m.kg.local_name, 'kg')
    self.assertEqual(kg.is_constant(), False)
    self.assertEqual(kg.is_fixed(), True)
    self.assertEqual(kg.is_parameter_type(), False)
    self.assertEqual(kg.is_variable_type(), False)
    self.assertEqual(kg.is_potentially_variable(), False)
    self.assertEqual(kg.is_named_expression_type(), False)
    self.assertEqual(kg.is_expression_type(), False)
    self.assertEqual(kg.is_component_type(), False)
    self.assertEqual(kg.is_expression_type(EXPR.ExpressionType.RELATIONAL), False)
    self.assertEqual(kg.is_indexed(), False)
    self.assertEqual(kg._compute_polynomial_degree(None), 0)
    with self.assertRaises(TypeError):
        x = float(kg)
    with self.assertRaises(TypeError):
        x = int(kg)
    assert_units_consistent(kg < m.kg)
    assert_units_consistent(kg > m.kg)
    assert_units_consistent(kg <= m.kg)
    assert_units_consistent(kg >= m.kg)
    assert_units_consistent(kg == m.kg)
    assert_units_consistent(kg + m.kg)
    assert_units_consistent(kg - m.kg)
    with self.assertRaises(InconsistentUnitsError):
        assert_units_consistent(kg + 3)
    with self.assertRaises(InconsistentUnitsError):
        assert_units_consistent(kg - 3)
    with self.assertRaises(InconsistentUnitsError):
        assert_units_consistent(3 + kg)
    with self.assertRaises(InconsistentUnitsError):
        assert_units_consistent(3 - kg)
    self.assertEqual(str(uc.get_units(kg * 3)), 'kg')
    self.assertEqual(str(uc.get_units(3 * kg)), 'kg')
    self.assertEqual(str(uc.get_units(kg / 3.0)), 'kg')
    self.assertEqual(str(uc.get_units(3.0 / kg)), '1/kg')
    self.assertEqual(str(uc.get_units(kg ** 2)), 'kg**2')
    x = 2 ** kg
    with self.assertRaises(UnitsError):
        assert_units_consistent(x)
    x = kg
    x += kg
    self.assertEqual(str(uc.get_units(x)), 'kg')
    x = kg
    x -= 2.0 * kg
    self.assertEqual(str(uc.get_units(x)), 'kg')
    x = kg
    x *= 3
    self.assertEqual(str(uc.get_units(x)), 'kg')
    x = kg
    x **= 3
    self.assertEqual(str(uc.get_units(x)), 'kg**3')
    self.assertEqual(str(uc.get_units(-kg)), 'kg')
    self.assertEqual(str(uc.get_units(+kg)), 'kg')
    self.assertEqual(str(uc.get_units(abs(kg))), 'kg')
    self.assertEqual(str(kg), 'kg')
    self.assertEqual(kg.to_string(), 'kg')
    self.assertEqual(kg.to_string(verbose=True), 'kg')
    self.assertEqual((kg / uc.s).to_string(), 'kg/s')
    self.assertEqual((kg * uc.m ** 2 / uc.s).to_string(), 'kg*m**2/s')
    m.v = Var(initialize=3, units=uc.J)
    e = uc.convert(m.v, uc.g * uc.m ** 2 / uc.s ** 2)
    self.assertEqual(e.to_string(), '1000.0*(g*m**2/J/s**2)*v')
    with self.assertRaisesRegex(PyomoException, 'Cannot convert non-constant Pyomo numeric value \\(kg\\) to bool.'):
        bool(kg)
    self.assertEqual(kg(), 1.0)
    self.assertEqual(value(kg), 1.0)
    buf = StringIO()
    kg.pprint(ostream=buf)
    self.assertEqual('kg', buf.getvalue())
    dless = uc.dimensionless
    self.assertEqual('dimensionless', str(dless))