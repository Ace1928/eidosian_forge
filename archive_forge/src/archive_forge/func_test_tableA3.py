from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
def test_tableA3(self):
    self.check_skiplist('tableA3')
    with capture_output(currdir + 'loadA3.dat'):
        print('load ' + self.filename('A') + ' format=set : A ;')
    model = AbstractModel()
    model.A = Set()
    instance = model.create_instance(currdir + 'loadA3.dat')
    self.assertEqual(set(instance.A.data()), set(['A1', 'A2', 'A3']))
    os.remove(currdir + 'loadA3.dat')