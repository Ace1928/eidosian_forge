import os
from io import StringIO
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.core.base import NumericLabeler, SymbolMap
from pyomo.environ import (
from pyomo.gdp import Disjunction
from pyomo.network import Port, Arc
from pyomo.repn.plugins.gams_writer import (
def test_nested_GDP_with_deactivate(self):
    m = ConcreteModel()
    m.x = Var(bounds=(0, 1))

    @m.Disjunct([0, 1])
    def disj(disj, _):

        @disj.Disjunct(['A', 'B'])
        def nested(n_disj, _):
            pass
        return disj
    m.choice = Disjunction(expr=[m.disj[0], m.disj[1]])
    m.c = Constraint(expr=m.x ** 2 + m.disj[1].nested['A'].indicator_var >= 1)
    m.disj[0].indicator_var.fix(1)
    m.disj[1].deactivate()
    m.disj[0].nested['A'].indicator_var.fix(1)
    m.disj[0].nested['B'].deactivate()
    m.disj[1].nested['A'].indicator_var.set_value(1)
    m.disj[1].nested['B'].deactivate()
    m.o = Objective(expr=m.x)
    TransformationFactory('gdp.fix_disjuncts').apply_to(m)
    outs = StringIO()
    m.write(outs, format='gams', io_options=dict(solver='dicopt'))
    self.assertIn('USING minlp', outs.getvalue())