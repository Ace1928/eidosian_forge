import os
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import ProblemFormat
from pyomo.environ import (
from ..nl_diff import load_and_compare_nl_baseline
def test_EXPORT_suffixes_int(self):
    model = ConcreteModel()
    model.junk = Suffix(direction=Suffix.EXPORT, datatype=Suffix.INT)
    model.junk_inactive = Suffix(direction=Suffix.EXPORT, datatype=Suffix.INT)
    model.x = Var()
    model.junk.set_value(model.x, 1)
    model.junk_inactive.set_value(model.x, 1)
    model.y = Var([1, 2], dense=True)
    model.junk.set_value(model.y, 2)
    model.junk_inactive.set_value(model.y, 2)
    model.obj = Objective(expr=model.x + sum_product(model.y))
    model.junk.set_value(model.obj, 3)
    model.junk_inactive.set_value(model.obj, 3)
    model.conx = Constraint(expr=model.x >= 1)
    model.junk.set_value(model.conx, 4)
    model.junk_inactive.set_value(model.conx, 4)
    model.cony = Constraint([1, 2], rule=lambda model, i: model.y[i] >= 1)
    model.junk.set_value(model.cony, 5)
    model.junk_inactive.set_value(model.cony, 5)
    model.junk.set_value(model, 6)
    model.junk_inactive.set_value(model, 6)
    model.junk_inactive.deactivate()
    _test = os.path.join(self.tempdir, 'EXPORT_suffixes.test.nl')
    model.write(filename=_test, format=self.nl_version, io_options={'symbolic_solver_labels': False, 'file_determinism': 1})
    _base = os.path.join(currdir, 'EXPORT_suffixes_int.baseline.nl')
    self.assertEqual(*load_and_compare_nl_baseline(_base, _test))