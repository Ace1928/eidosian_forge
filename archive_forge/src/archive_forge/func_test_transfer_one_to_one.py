import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.mpc.interfaces.var_linker import DynamicVarLinker
def test_transfer_one_to_one(self):
    m1, m2 = self._make_models()
    vars1 = [pyo.Reference(m1.var[:, 'A']), pyo.Reference(m1.var[:, 'B']), m1.input]
    vars2 = [m2.x1, m2.x2, m2.x3]
    linker = DynamicVarLinker(vars1, vars2)
    t_source = 0
    t_target = 2
    linker.transfer(t_source=0, t_target=2)
    pred_AB = lambda t: 1.0 + t * 0.1
    pred_input = lambda t: 1.0 - t * 0.1
    for t in m1.time:
        self.assertEqual(m1.var[t, 'A'].value, pred_AB(t))
        self.assertEqual(m1.var[t, 'B'].value, pred_AB(t))
        self.assertEqual(m1.input[t].value, pred_input(t))
        if t == t_target:
            self.assertEqual(m2.x1[t].value, pred_AB(t_source))
            self.assertEqual(m2.x2[t].value, pred_AB(t_source))
            self.assertEqual(m2.x3[t].value, pred_input(t_source))
            self.assertEqual(m2.x4[t].value, 2.4)
        else:
            self.assertEqual(m2.x1[t].value, 2.1)
            self.assertEqual(m2.x2[t].value, 2.2)
            self.assertEqual(m2.x3[t].value, 2.3)
            self.assertEqual(m2.x4[t].value, 2.4)