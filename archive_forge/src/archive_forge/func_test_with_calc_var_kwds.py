import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.dependencies import networkx_available
from pyomo.common.dependencies import scipy_available
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.contrib.incidence_analysis.scc_solver import (
from pyomo.contrib.incidence_analysis.tests.models_for_testing import (
import pyomo.common.unittest as unittest
def test_with_calc_var_kwds(self):
    m = pyo.ConcreteModel()
    m.v0 = pyo.Var()
    m.v1 = pyo.Var()
    m.v2 = pyo.Var(initialize=79703634.05074187)
    m.v2.fix()
    m.p0 = pyo.Param(initialize=300000.0)
    m.p1 = pyo.Param(initialize=1296000000000.0)
    m.con0 = pyo.Constraint(expr=m.v0 == m.p0)
    m.con1 = pyo.Constraint(expr=0.0 == m.p1 * m.v1 / m.v0 + m.v2)
    calc_var_kwds = {'eps': 1e-07}
    results = solve_strongly_connected_components(m, calc_var_kwds=calc_var_kwds)
    self.assertEqual(len(results), 2)
    self.assertAlmostEqual(m.v0.value, m.p0.value)
    self.assertAlmostEqual(m.v1.value, -18.4499152895)