from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
def test_md12(self):
    model = ConcreteModel()
    model.A = Set()
    md = DataPortal()
    try:
        md.load(filename=example_dir + 'A.tab', format='bad', set=model.A)
        self.fail('Bad format error')
    except ValueError:
        pass
    try:
        md.load(filename=example_dir + 'A.tab')
        self.fail('Bad format error')
    except ValueError:
        pass