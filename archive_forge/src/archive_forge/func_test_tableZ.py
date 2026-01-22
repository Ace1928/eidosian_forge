from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
def test_tableZ(self):
    with capture_output(currdir + 'loadZ.dat'):
        print('table Z := 1.01 ;')
    model = AbstractModel()
    model.Z = Param(default=99.0)
    instance = model.create_instance(currdir + 'loadZ.dat')
    self.assertEqual(instance.Z, 1.01)
    os.remove(currdir + 'loadZ.dat')