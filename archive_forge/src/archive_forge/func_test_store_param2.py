from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
def test_store_param2(self):
    self.check_skiplist('store_param2')
    model = ConcreteModel()
    model.A = Set(initialize=[1, 2, 3])
    model.p = Param(model.A, initialize={1: 10, 2: 20, 3: 30})
    data = DataPortal()
    data.store(param=model.p, **self.create_write_options('param2'))
    self.compare_data('param2', self.suffix)