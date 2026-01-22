import json
import pickle
import os
from os.path import abspath, dirname, join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
import pyomo.opt
from pyomo.common.dependencies import yaml, yaml_available
def test_soln_pprint1(self):
    """Write a solution with only zero values, using the results 'write()' method"""
    self.soln.variable[1]['Value'] = 0.0
    self.soln.variable[2]['Value'] = 0.0
    self.soln.variable[4]['Value'] = 0.0
    self.results.write(filename=join(currdir, 'soln_pprint.txt'))
    if not os.path.exists(join(currdir, 'soln_pprint.txt')):
        self.fail('test_write_solution - failed to write soln_pprint.txt')
    _out, _log = (join(currdir, 'soln_pprint.txt'), join(currdir, 'test3_soln.txt'))
    self.assertTrue(cmp(_out, _log), msg='Files %s and %s differ' % (_out, _log))