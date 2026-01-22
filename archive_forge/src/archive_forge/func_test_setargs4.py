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
def test_setargs4(self):
    model = ConcreteModel()
    model.A = Set(initialize=[1])
    model.B = Set(model.A, initialize={1: [1]})
    try:
        model.C = Set(model.B)
        self.fail('test_setargs4 - expected error when passing in a set that is indexed')
    except TypeError:
        pass