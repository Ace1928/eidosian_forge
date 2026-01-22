import os
import pyomo.common.unittest as unittest
from pyomo.common.gsl import find_GSL
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
from pyomo.environ import (
from ..nl_diff import load_and_compare_nl_baseline
import pyomo.repn.plugins.ampl.ampl_ as ampl_
import pyomo.repn.plugins.nl_writer as nl_writer
def test_var_on_nonblock(self):
    if self._nl_version != 'nl_v1':
        self.skipTest(f'test not applicable to writer {self._nl_version}')

    class Foo(Block().__class__):

        def __init__(self, *args, **kwds):
            kwds.setdefault('ctype', Foo)
            super(Foo, self).__init__(*args, **kwds)
    model = ConcreteModel()
    model.x = Var()
    model.other = Foo()
    model.other.a = Var()
    model.c = Constraint(expr=model.other.a + 2 * model.x <= 0)
    model.obj = Objective(expr=model.x)
    baseline_fname, test_fname = self._get_fnames()
    self.assertRaisesRegex(KeyError, "'other.a' exists within Foo 'other'", model.write, test_fname, format=self._nl_version)