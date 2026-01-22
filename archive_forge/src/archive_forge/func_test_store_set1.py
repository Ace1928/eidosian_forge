from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
def test_store_set1(self):
    self.check_skiplist('store_set1')
    model = ConcreteModel()
    model.A = Set(initialize=[1, 3, 5])
    data = DataPortal()
    data.store(set=model.A, **self.create_write_options('set1'))
    self.compare_data('set1', self.suffix)