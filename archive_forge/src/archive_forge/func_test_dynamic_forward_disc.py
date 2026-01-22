import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.dependencies import networkx_available
from pyomo.common.dependencies import scipy_available
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.contrib.incidence_analysis.scc_solver import (
from pyomo.contrib.incidence_analysis.tests.models_for_testing import (
import pyomo.common.unittest as unittest
def test_dynamic_forward_disc(self):
    nfe = 5
    m = make_dynamic_model(nfe=nfe, scheme='FORWARD')
    time = m.time
    t0 = m.time.first()
    t1 = m.time.next(t0)
    m.flow_in.fix()
    m.height[t0].fix()
    constraints = list(m.component_data_objects(pyo.Constraint))
    self.assertEqual(len(list(generate_strongly_connected_components(constraints))), len(list(m.component_data_objects(pyo.Constraint))))
    self.assertEqual(len(list(generate_strongly_connected_components(constraints))), 3 * nfe + 2)
    for i, (block, inputs) in enumerate(generate_strongly_connected_components(constraints)):
        with TemporarySubsystemManager(to_fix=inputs):
            idx = i // 3
            mod = i % 3
            t = m.time[idx + 1]
            if t != time.last():
                t_next = m.time.next(t)
            self.assertEqual(len(block.vars), 1)
            self.assertEqual(len(block.cons), 1)
            if mod == 0:
                self.assertIs(block.vars[0], m.flow_out[t])
                self.assertIs(block.cons[0], m.flow_out_eqn[t])
            elif mod == 1:
                self.assertIs(block.vars[0], m.dhdt[t])
                self.assertIs(block.cons[0], m.diff_eqn[t])
            elif mod == 2:
                self.assertIs(block.vars[0], m.height[t_next])
                self.assertIs(block.cons[0], m.dhdt_disc_eq[t])