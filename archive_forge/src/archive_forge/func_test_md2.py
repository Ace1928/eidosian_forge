from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
def test_md2(self):
    md = DataPortal()
    model = AbstractModel()
    model.A = Set()
    md.load(model=model, filename=currdir + 'data1.dat')
    self.assertEqual(set(md['A']), set([1, 2, 3]))