from io import StringIO
import logging
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import _parent_disjunct
from pyomo.common.log import LoggingIntercept
def test_bounds_flipped_for_negative_coefs(self):
    m = ConcreteModel()
    m.Time = Set(initialize=range(4))
    m.makespan = Var(bounds=(0, 4))
    m.act_time = Var(m.Time, domain=Binary)
    m.cost = Var(bounds=(1, 6))

    @m.Disjunct()
    def d1(d):
        d.c = Constraint(expr=m.cost == 6)

    @m.Disjunct()
    def d2(d):
        d.c = Constraint(expr=m.cost == 5)
        d.do_act = Constraint(expr=sum((m.act_time[t] for t in m.Time)) == 1)

        @d.Constraint(m.Time)
        def ms(d, t):
            return t * m.act_time[t] + 2 <= m.makespan
    m.disjunction = Disjunction(expr=[m.d1, m.d2])
    m.obj = Objective(expr=m.cost)
    m.act_time.fix(0)
    m.act_time[2].unfix()
    bt = TransformationFactory('gdp.bound_pretransformation')
    bt.apply_to(m)
    cons = bt.get_transformed_constraints(m.act_time[2], m.disjunction)
    self.assertEqual(len(cons), 2)
    lb = cons[0]
    assertExpressionsEqual(self, lb.expr, 0 * m.disjunction.disjuncts[0].binary_indicator_var + m.disjunction.disjuncts[1].binary_indicator_var <= m.act_time[2])
    ub = cons[1]
    assertExpressionsEqual(self, ub.expr, m.disjunction.disjuncts[0].binary_indicator_var + m.disjunction.disjuncts[1].binary_indicator_var >= m.act_time[2])
    cons = bt.get_transformed_constraints(m.act_time[0], m.disjunction)
    self.assertEqual(len(cons), 0)
    cons = bt.get_transformed_constraints(m.act_time[1], m.disjunction)
    self.assertEqual(len(cons), 0)
    cons = bt.get_transformed_constraints(m.act_time[3], m.disjunction)
    self.assertEqual(len(cons), 0)
    cons = bt.get_transformed_constraints(m.makespan, m.disjunction)
    self.assertEqual(len(cons), 2)
    lb = cons[0]
    assertExpressionsEqual(self, lb.expr, 0 * m.disjunction.disjuncts[0].binary_indicator_var + 2.0 * m.disjunction.disjuncts[1].binary_indicator_var <= m.makespan)
    ub = cons[1]
    assertExpressionsEqual(self, ub.expr, 4 * m.disjunction.disjuncts[0].binary_indicator_var + 4 * m.disjunction.disjuncts[1].binary_indicator_var >= m.makespan)
    cons = bt.get_transformed_constraints(m.cost, m.disjunction)
    self.assertEqual(len(cons), 2)
    lb = cons[0]
    assertExpressionsEqual(self, lb.expr, 6.0 * m.disjunction.disjuncts[0].binary_indicator_var + 5.0 * m.disjunction.disjuncts[1].binary_indicator_var <= m.cost)
    ub = cons[1]
    assertExpressionsEqual(self, ub.expr, 6 * m.disjunction.disjuncts[0].binary_indicator_var + 5.0 * m.disjunction.disjuncts[1].binary_indicator_var >= m.cost)