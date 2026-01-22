import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.network import Port, Arc
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.gdp import Disjunct, Disjunction
from pyomo.core.base.units_container import pint_available, UnitsError
from pyomo.util.check_units import (
def test_assert_units_consistent_on_datas(self):
    u = units
    m = ConcreteModel()
    m.S = Set(initialize=[1, 2, 3])
    m.x = Var(m.S, units=u.m)
    m.t = Var(m.S, units=u.s)
    m.v = Var(m.S, units=u.m / u.s)
    m.unitless = Var(m.S)

    @m.Constraint(m.S)
    def vel_con(m, i):
        return m.v[i] == m.x[i] / m.t[i]

    @m.Constraint(m.S)
    def unitless_con(m, i):
        return m.unitless[i] == 42.0

    @m.Constraint(m.S)
    def sqrt_con(m, i):
        return sqrt(m.v[i]) == sqrt(m.x[i] / m.t[i])
    assert_units_consistent(m)
    assert_units_consistent(m.x)
    assert_units_consistent(m.t)
    assert_units_consistent(m.v)
    assert_units_consistent(m.unitless)
    assert_units_consistent(m.vel_con)
    assert_units_consistent(m.unitless_con)
    assert_units_consistent(m.x[2])
    assert_units_consistent(m.t[2])
    assert_units_consistent(m.v[2])
    assert_units_consistent(m.unitless[2])
    assert_units_consistent(m.vel_con[2])
    assert_units_consistent(m.unitless_con[2])
    assert_units_equivalent(m.x[2], m.x[1])
    assert_units_equivalent(m.t[2], u.s)
    assert_units_equivalent(m.v[2], u.m / u.s)
    assert_units_equivalent(m.unitless[2], u.dimensionless)
    assert_units_equivalent(m.unitless[2], None)
    assert_units_equivalent(m.vel_con[2].body, u.m / u.s)
    assert_units_equivalent(m.unitless_con[2].body, u.dimensionless)

    @m.Constraint(m.S)
    def broken(m, i):
        return m.x[i] == 42.0 * m.v[i]
    with self.assertRaises(UnitsError):
        assert_units_consistent(m)
    with self.assertRaises(UnitsError):
        assert_units_consistent(m.broken)
    with self.assertRaises(UnitsError):
        assert_units_consistent(m.broken[1])
    assert_units_consistent(m.x)
    assert_units_consistent(m.t)
    assert_units_consistent(m.v)
    assert_units_consistent(m.unitless)
    assert_units_consistent(m.vel_con)
    assert_units_consistent(m.unitless_con)
    assert_units_consistent(m.x[2])
    assert_units_consistent(m.t[2])
    assert_units_consistent(m.v[2])
    assert_units_consistent(m.unitless[2])
    assert_units_consistent(m.vel_con[2])
    assert_units_consistent(m.unitless_con[2])