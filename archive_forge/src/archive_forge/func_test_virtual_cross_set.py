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
def test_virtual_cross_set(self):
    self.model.C = self.model.A * self.model.B
    self.model.C.virtual = True
    self.instance = self.model.create_instance()
    self.assertEqual(len(self.instance.C), 9)
    if self.instance.C.value is not None:
        self.assertEqual(len(self.instance.C.value), 9)
    tmp = []
    for item in self.instance.C:
        tmp.append(item)
    self.assertEqual(len(tmp), 9)