import json
import pickle
import os
from os.path import abspath, dirname, join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
import pyomo.opt
from pyomo.common.dependencies import yaml, yaml_available
def test_soln_suffix_getiter(self):
    self.soln.variable[1]['Value'] = 0.0
    self.soln.variable[2]['Value'] = 0.1
    self.soln.variable[4]['Value'] = 0.3
    self.assertEqual(self.soln.variable[4]['Value'], 0.3)
    self.assertEqual(self.soln.variable[2]['Value'], 0.1)