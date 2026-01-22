import copy
import itertools
import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
import pyomo.core.base
from pyomo.core.base.util import flatten_tuple
from pyomo.environ import (
from pyomo.core.base.set import _AnySet, RangeDifferenceError
def test_numpy_int(self):
    model = ConcreteModel()
    model.A = Set(initialize=[numpy.int_(1.0), numpy.int_(0.0)])
    self.assertEqual(model.A.bounds(), (0, 1))