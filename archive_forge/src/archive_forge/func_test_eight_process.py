import logging
from os.path import abspath, dirname, join, normpath
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.core import Binary, Block, ConcreteModel, Constraint, Integers, Var
from pyomo.gdp import Disjunct, Disjunction
from pyomo.util.model_size import build_model_size_report, log_model_size_report
from pyomo.common.fileutils import import_file
def test_eight_process(self):
    """Test with the eight process problem model."""
    exfile = import_file(join(exdir, 'eight_process', 'eight_proc_model.py'))
    eight_process = exfile.build_eight_process_flowsheet()
    model_size = build_model_size_report(eight_process)
    self.assertEqual(model_size.activated.variables, 44)
    self.assertEqual(model_size.overall.variables, 44)
    self.assertEqual(model_size.activated.binary_variables, 12)
    self.assertEqual(model_size.overall.binary_variables, 12)
    self.assertEqual(model_size.activated.integer_variables, 0)
    self.assertEqual(model_size.overall.integer_variables, 0)
    self.assertEqual(model_size.activated.constraints, 52)
    self.assertEqual(model_size.overall.constraints, 52)
    self.assertEqual(model_size.activated.disjuncts, 12)
    self.assertEqual(model_size.overall.disjuncts, 12)
    self.assertEqual(model_size.activated.disjunctions, 5)
    self.assertEqual(model_size.overall.disjunctions, 5)