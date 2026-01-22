from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
def test_md18(self):
    cwd = os.getcwd()
    os.chdir(currdir)
    md = DataPortal()
    md.load(filename=currdir + 'data17.dat')
    self.assertEqual(md['A'], 1)
    self.assertEqual(md['B'], {'a': 1})
    self.assertEqual(md['C'], {'a': 1, 'b': 2, 'c': 3})
    self.assertEqual(md['D'], 1)
    os.chdir(cwd)