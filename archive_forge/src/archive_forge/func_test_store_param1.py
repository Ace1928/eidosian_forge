from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
def test_store_param1(self):
    self.check_skiplist('store_param1')
    model = ConcreteModel()
    model.p = Param(initialize=1)
    data = DataPortal()
    data.store(param=model.p, **self.create_write_options('param1'))
    self.compare_data('param1', self.suffix)