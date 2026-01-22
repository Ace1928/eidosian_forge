from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
def test_tableXW_4(self):
    self.check_skiplist('tableXW_4')
    with capture_output(currdir + 'loadXW.dat'):
        print('load ' + self.filename('XW') + ' : B=[A] R=X S=W;')
    model = AbstractModel()
    model.B = Set()
    model.R = Param(model.B)
    model.S = Param(model.B)
    instance = model.create_instance(currdir + 'loadXW.dat')
    self.assertEqual(set(instance.B.data()), set(['A1', 'A2', 'A3']))
    self.assertEqual(instance.R.extract_values(), {'A1': 3.3, 'A2': 3.4, 'A3': 3.5})
    self.assertEqual(instance.S.extract_values(), {'A1': 4.3, 'A2': 4.4, 'A3': 4.5})
    os.remove(currdir + 'loadXW.dat')