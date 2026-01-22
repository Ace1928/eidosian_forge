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
def test_initialize6(self):
    self.model.A = Set(initialize=range(0, 4))

    def B_index(model):
        for i in model.A:
            if i % 2 == 0:
                yield i

    def B_init(model, i, j):
        k = i + j
        if j:
            return range(i, 2 + i)
        return []
    self.model.B = Set(B_index, [True, False], initialize=B_init)
    self.instance = self.model.create_instance()
    self.assertEqual(set(self.instance.B.keys()), set([(0, True), (2, True), (0, False), (2, False)]))
    self.assertEqual(self.instance.B[0, True].value, set([0, 1]))
    self.assertEqual(self.instance.B[2, True].value, set([2, 3]))