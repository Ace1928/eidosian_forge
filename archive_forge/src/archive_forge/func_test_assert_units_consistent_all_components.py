import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.network import Port, Arc
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.gdp import Disjunct, Disjunction
from pyomo.core.base.units_container import pint_available, UnitsError
from pyomo.util.check_units import (
def test_assert_units_consistent_all_components(self):
    u = units
    m = self._create_model_and_vars()
    m.obj = Objective(expr=m.dx / m.t - m.vx)
    m.con = Constraint(expr=m.dx / m.t == m.vx)
    m.exp = Expression(expr=m.dx / m.t - m.vx)
    m.suff = Suffix(direction=Suffix.LOCAL)
    m.rs = RangeSet(5)
    m.disj1 = Disjunct()
    m.disj1.constraint = Constraint(expr=m.dx / m.t <= m.vx)
    m.disj2 = Disjunct()
    m.disj2.constraint = Constraint(expr=m.dx / m.t <= m.vx)
    m.disjn = Disjunction(expr=[m.disj1, m.disj2])
    m.extfn = ExternalFunction(python_callback_function, units=u.m / u.s, arg_units=[u.m, u.s])
    m.conext = Constraint(expr=m.extfn(m.dx, m.t) - m.vx == 0)
    m.cset = ContinuousSet(bounds=(0, 1))
    m.svar = Var(m.cset, units=u.m)
    m.dvar = DerivativeVar(sVar=m.svar, units=u.m / u.s)

    def prt1_rule(m):
        return {'avar': m.dx}

    def prt2_rule(m):
        return {'avar': m.dy}
    m.prt1 = Port(rule=prt1_rule)
    m.prt2 = Port(rule=prt2_rule)

    def arcrule(m):
        return dict(source=m.prt1, destination=m.prt2)
    m.arc = Arc(rule=arcrule)
    assert_units_consistent(m)