import platform
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import find_library
from pyomo.opt import check_available_solvers
from pyomo.core.base.external import nan
@unittest.skipIf(is_pypy, 'Cannot evaluate external functions under pypy')
def test_eval_function_fgh(self):
    m = pyo.ConcreteModel()
    m.tf = pyo.ExternalFunction(library=flib, function='demo_function')
    f, g, h = m.tf.evaluate_fgh(('sum', 1, 2, 3))
    self.assertEqual(f, 6)
    self.assertEqual(g, [nan, 1, 1, 1])
    self.assertEqual(h, [nan, nan, 0, nan, 0, 0, nan, 0, 0, 0])
    f, g, h = m.tf.evaluate_fgh(('inv', 1, 2, 3))
    self.assertAlmostEqual(f, 1.8333333, 4)
    self.assertStructuredAlmostEqual(g, [nan, -1, -1 / 4, -1 / 9])
    self.assertStructuredAlmostEqual(h, [nan, nan, 2, nan, 0, 1 / 4, nan, 0, 0, 2 / 27])