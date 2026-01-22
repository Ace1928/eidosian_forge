import copy
import itertools
import logging
import pickle
from io import StringIO
from collections import namedtuple as NamedTuple
import pyomo.common.unittest as unittest
from pyomo.common import DeveloperError
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import pandas as pd, pandas_available
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr import native_numeric_types, native_types
import pyomo.core.base.set as SetModule
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.initializer import (
from pyomo.core.base.set import (
from pyomo.environ import (
def test_ordered_nondim_setproduct(self):
    NonDim = Set(initialize=[2, (2, 3)], dimen=None)
    NonDim.construct()
    NonDim2 = Set(initialize=[4, (3, 4)], dimen=None)
    NonDim2.construct()
    x = SetOf([1]).cross(NonDim, SetOf([3, 4, 5]))
    self.assertEqual(len(x), 6)
    try:
        origFlattenCross = SetModule.FLATTEN_CROSS_PRODUCT
        SetModule.FLATTEN_CROSS_PRODUCT = True
        ref = [(1, 2, 3), (1, 2, 4), (1, 2, 5), (1, 2, 3, 3), (1, 2, 3, 4), (1, 2, 3, 5)]
        self.assertEqual(list(x), ref)
        self.assertEqual(x.dimen, None)
        SetModule.FLATTEN_CROSS_PRODUCT = False
        ref = [(1, 2, 3), (1, 2, 4), (1, 2, 5), (1, (2, 3), 3), (1, (2, 3), 4), (1, (2, 3), 5)]
        self.assertEqual(list(x), ref)
        self.assertEqual(x.dimen, 3)
    finally:
        SetModule.FLATTEN_CROSS_PRODUCT = origFlattenCross
    self.assertIn((1, 2, 3), x)
    self.assertNotIn((1, 2, 6), x)
    self.assertIn((1, (2, 3), 3), x)
    self.assertIn((1, 2, 3, 3), x)
    self.assertNotIn((1, (2, 4), 3), x)
    self.assertEqual(x.ord((1, 2, 3)), 1)
    self.assertEqual(x.ord((1, (2, 3), 3)), 4)
    self.assertEqual(x.ord((1, (2, 3), 5)), 6)
    self.assertEqual(x.ord((1, 2, 3, 3)), 4)
    self.assertEqual(x.ord((1, 2, 3, 5)), 6)
    x = SetOf([1]).cross(NonDim, NonDim2, SetOf([0, 1]))
    self.assertEqual(len(x), 8)
    try:
        origFlattenCross = SetModule.FLATTEN_CROSS_PRODUCT
        SetModule.FLATTEN_CROSS_PRODUCT = True
        ref = [(1, 2, 4, 0), (1, 2, 4, 1), (1, 2, 3, 4, 0), (1, 2, 3, 4, 1), (1, 2, 3, 4, 0), (1, 2, 3, 4, 1), (1, 2, 3, 3, 4, 0), (1, 2, 3, 3, 4, 1)]
        self.assertEqual(list(x), ref)
        for i, v in enumerate(ref):
            self.assertEqual(x[i + 1], v)
        self.assertEqual(x.dimen, None)
        SetModule.FLATTEN_CROSS_PRODUCT = False
        ref = [(1, 2, 4, 0), (1, 2, 4, 1), (1, 2, (3, 4), 0), (1, 2, (3, 4), 1), (1, (2, 3), 4, 0), (1, (2, 3), 4, 1), (1, (2, 3), (3, 4), 0), (1, (2, 3), (3, 4), 1)]
        self.assertEqual(list(x), ref)
        for i, v in enumerate(ref):
            self.assertEqual(x[i + 1], v)
        self.assertEqual(x.dimen, 4)
    finally:
        SetModule.FLATTEN_CROSS_PRODUCT = origFlattenCross
    self.assertIn((1, 2, 4, 0), x)
    self.assertNotIn((1, 2, 6), x)
    self.assertIn((1, (2, 3), 4, 0), x)
    self.assertIn((1, 2, (3, 4), 0), x)
    self.assertIn((1, 2, 3, 4, 0), x)
    self.assertNotIn((1, 2, 5, 4, 0), x)
    self.assertEqual(x.ord((1, 2, 4, 0)), 1)
    self.assertEqual(x.ord((1, (2, 3), 4, 0)), 5)
    self.assertEqual(x.ord((1, 2, (3, 4), 0)), 3)
    self.assertEqual(x.ord((1, 2, 3, 4, 0)), 3)