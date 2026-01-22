import os.path
from pyomo.common.fileutils import this_file_dir, import_file
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.dependencies import attempt_import
from pyomo.common.log import LoggingIntercept
from pyomo.opt import TerminationCondition
from io import StringIO
import logging
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from pyomo.contrib.pynumero.asl import AmplInterface
import pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver as cyipopt_solver
import cyipopt as cyipopt_core
@unittest.skipIf(not pandas_available, 'pandas needed to run this example')
def test_cyipopt_functor(self):
    ex = import_file(os.path.join(example_dir, 'callback', 'cyipopt_functor_callback.py'))
    df = ex.main()
    self.assertEqual(df.shape, (7, 5))
    s = df['ca_bal']
    self.assertAlmostEqual(s.iloc[6], 0, places=3)