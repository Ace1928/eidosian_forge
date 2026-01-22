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
def test_external_grey_box_react_example_maximize_cb_outputs_scaling(self):
    ex = import_file(os.path.join(example_dir, 'external_grey_box', 'react_example', 'maximize_cb_ratio_residuals.py'))
    aoptions = {'nlp_scaling_method': 'user-scaling', 'output_file': '_cyipopt-external-greybox-react-scaling.log', 'file_print_level': 10}
    m = ex.maximize_cb_ratio_residuals_with_output_scaling(additional_options=aoptions)
    self.assertAlmostEqual(pyo.value(m.reactor.inputs['sv']), 1.26541996, places=3)
    self.assertAlmostEqual(pyo.value(m.reactor.inputs['cb']), 1071.7410089, places=2)
    self.assertAlmostEqual(pyo.value(m.reactor.outputs['cb_ratio']), 0.15190409266, places=3)
    with open('_cyipopt-external-greybox-react-scaling.log', 'r') as fd:
        solver_trace = fd.read()
    os.remove('_cyipopt-external-greybox-react-scaling.log')
    self.assertIn('nlp_scaling_method = user-scaling', solver_trace)
    self.assertIn('output_file = _cyipopt-external-greybox-react-scaling.log', solver_trace)
    self.assertIn('objective scaling factor = 1', solver_trace)
    self.assertIn('x scaling provided', solver_trace)
    self.assertIn('c scaling provided', solver_trace)
    self.assertIn('d scaling provided', solver_trace)
    self.assertIn('DenseVector "x scaling vector" with 7 elements:', solver_trace)
    self.assertIn('x scaling vector[    2]= 1.2000000000000000e+00', solver_trace)
    self.assertIn('x scaling vector[    7]= 1.7000000000000000e+00', solver_trace)
    self.assertIn('x scaling vector[    6]= 1.1000000000000001e+00', solver_trace)
    self.assertIn('x scaling vector[    1]= 1.3000000000000000e+00', solver_trace)
    self.assertIn('x scaling vector[    3]= 1.3999999999999999e+00', solver_trace)
    self.assertIn('x scaling vector[    4]= 1.5000000000000000e+00', solver_trace)
    self.assertIn('x scaling vector[    5]= 1.6000000000000001e+00', solver_trace)
    self.assertIn('DenseVector "c scaling vector" with 6 elements:', solver_trace)
    self.assertIn('c scaling vector[    1]= 4.2000000000000000e+01', solver_trace)
    self.assertIn('c scaling vector[    2]= 1.0000000000000001e-01', solver_trace)
    self.assertIn('c scaling vector[    3]= 2.0000000000000001e-01', solver_trace)
    self.assertIn('c scaling vector[    4]= 2.9999999999999999e-01', solver_trace)
    self.assertIn('c scaling vector[    5]= 4.0000000000000002e-01', solver_trace)
    self.assertIn('c scaling vector[    6]= 1.0000000000000000e+01', solver_trace)