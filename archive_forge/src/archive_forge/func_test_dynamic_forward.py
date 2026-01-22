import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.dependencies import networkx_available
from pyomo.common.dependencies import scipy_available
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.contrib.incidence_analysis.scc_solver import (
from pyomo.contrib.incidence_analysis.tests.models_for_testing import (
import pyomo.common.unittest as unittest
def test_dynamic_forward(self):
    nfe = 5
    m = make_dynamic_model(nfe=nfe, scheme='FORWARD')
    time = m.time
    t0 = time.first()
    m.flow_in.fix()
    m.height[t0].fix()
    solve_strongly_connected_components(m)
    for con in m.component_data_objects(pyo.Constraint):
        self.assertEqual(pyo.value(con.upper), pyo.value(con.lower))
        self.assertAlmostEqual(pyo.value(con.body), pyo.value(con.upper), delta=1e-07)
    for t in time:
        if t == t0:
            self.assertTrue(m.height[t].fixed)
        else:
            self.assertFalse(m.height[t].fixed)
        self.assertFalse(m.flow_out[t].fixed)
        self.assertFalse(m.dhdt[t].fixed)
        self.assertTrue(m.flow_in[t].fixed)