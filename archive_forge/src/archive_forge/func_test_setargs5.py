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
def test_setargs5(self):
    model = AbstractModel()
    model.A = Set()
    model.B = Set()
    model.C = model.A | model.B
    model.Z = Set(model.C)
    model.Y = RangeSet(model.C)
    model.X = Param(model.C, default=0.0)