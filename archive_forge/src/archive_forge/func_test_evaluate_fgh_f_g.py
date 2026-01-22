import os
import shutil
import sys
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.gsl import find_GSL
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.environ import (
from pyomo.core.base.external import (
from pyomo.core.base.units_container import pint_available, units
from pyomo.core.expr.numeric_expr import (
from pyomo.opt import check_available_solvers
def test_evaluate_fgh_f_g(self):
    m = ConcreteModel()
    m.f = ExternalFunction(_f, _g)
    with self.assertRaisesRegex(RuntimeError, "ExternalFunction 'f' was not defined with a Hessian callback."):
        f, g, h = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id))
    f, g, h = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id), fgh=1)
    self.assertEqual(f, 5 ** 2 + 3 * 5 * 7 + 5 * 7 * 11 ** 2)
    self.assertEqual(g, [2 * 5 + 3 * 7 + 7 * 11 ** 2, 3 * 5 + 5 * 11 ** 2, 2 * 5 * 7 * 11, 0])
    self.assertIsNone(h)
    f, g, h = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id), fgh=0)
    self.assertEqual(f, 5 ** 2 + 3 * 5 * 7 + 5 * 7 * 11 ** 2)
    self.assertIsNone(g)
    self.assertIsNone(h)