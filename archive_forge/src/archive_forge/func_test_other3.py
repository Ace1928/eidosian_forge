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
def test_other3(self):

    def tmp_init(model, i):
        tmp = []
        for i in range(0, value(model.n)):
            tmp.append(i / 2.0)
        return tmp
    self.model.n = Param(initialize=5)
    self.model.Z = Set(initialize=['A'])
    self.model.A = Set(self.model.Z, initialize=tmp_init, validate=lambda model, x: x in Integers)
    try:
        self.instance = self.model.create_instance()
    except ValueError:
        pass
    else:
        self.fail('fail test_other1')