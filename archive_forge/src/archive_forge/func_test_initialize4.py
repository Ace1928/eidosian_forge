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
def test_initialize4(self):
    self.model.A = Set(initialize=range(0, 4))

    def B_index(model):
        return (i for i in model.A if i % 2 == 0)

    def B_init(model, i):
        return range(i, 2 + i)
    self.model.B = Set(B_index, initialize=B_init)
    self.instance = self.model.create_instance()
    self.assertEqual(self.instance.B[0].value, set([0, 1]))
    self.assertEqual(self.instance.B[2].value, set([2, 3]))
    self.assertEqual(list(sorted(self.instance.B.keys())), [0, 2])