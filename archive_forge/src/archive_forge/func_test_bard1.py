import os
from os.path import abspath, dirname, normpath, join
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
import pyomo.opt
import pyomo.scripting.pyomo_main as pyomo_main
from pyomo.scripting.util import cleanup
def test_bard1(self):
    self.problem = 'test_bard1'
    self.run_solver(join(exdir, 'bard1.py'))
    self.check('bard1', self.solver)