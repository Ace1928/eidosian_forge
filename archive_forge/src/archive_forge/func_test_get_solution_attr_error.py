import json
import pickle
import os
from os.path import abspath, dirname, join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
import pyomo.opt
from pyomo.common.dependencies import yaml, yaml_available
def test_get_solution_attr_error(self):
    """Create an error with a solution suffix"""
    try:
        tmp = self.soln.bad
        self.fail("Expected attribute error failure for 'bad'")
    except AttributeError:
        pass