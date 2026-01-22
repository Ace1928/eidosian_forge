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
def test_arrayset_construct(self):

    def tmp_constructor(model, ctr, index):
        if ctr == 10:
            return Set.End
        else:
            return ctr
    a = Set(initialize=[1, 2, 3])
    a.construct()
    b = Set(a, initialize=tmp_constructor)
    try:
        b.construct({4: None})
        self.fail('test_arrayset_construct - expected KeyError')
    except KeyError:
        pass
    b._constructed = False
    b.construct()
    self.assertEqual(len(b), 3)
    for i in b:
        self.assertEqual(i in a, True)
    self.assertEqual(b[1], [1, 2, 3, 4, 5, 6, 7, 8, 9])
    self.assertEqual(b[2], [1, 2, 3, 4, 5, 6, 7, 8, 9])
    self.assertEqual(b[3], [1, 2, 3, 4, 5, 6, 7, 8, 9])
    b = Set(a, a, initialize=tmp_constructor)
    with self.assertRaisesRegex(TypeError, "'int' object is not iterable"):
        b.construct()