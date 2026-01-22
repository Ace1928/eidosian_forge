import json
import pickle
import os
from os.path import abspath, dirname, join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
import pyomo.opt
from pyomo.common.dependencies import yaml, yaml_available
def test_write_solution1(self):
    """Write a SolverResults Object with solutions"""
    self.results.write(filename=join(currdir, 'write_solution1.txt'))
    if not os.path.exists(join(currdir, 'write_solution1.txt')):
        self.fail('test_write_solution - failed to write write_solution1.txt')
    _log, _out = (join(currdir, 'write_solution1.txt'), join(currdir, 'test1_soln.txt'))
    self.assertTrue(cmp(_out, _log), msg='Files %s and %s differ' % (_out, _log))