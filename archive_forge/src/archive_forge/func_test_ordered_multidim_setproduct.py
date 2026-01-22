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
def test_ordered_multidim_setproduct(self):
    x = SetOf([(1, 2), (3, 4)]) * SetOf([(5, 6), (7, 8)])
    self.assertEqual(x.dimen, 4)
    try:
        origFlattenCross = SetModule.FLATTEN_CROSS_PRODUCT
        SetModule.FLATTEN_CROSS_PRODUCT = True
        ref = [(1, 2, 5, 6), (1, 2, 7, 8), (3, 4, 5, 6), (3, 4, 7, 8)]
        self.assertEqual(list(x), ref)
        self.assertEqual(x.dimen, 4)
        SetModule.FLATTEN_CROSS_PRODUCT = False
        ref = [((1, 2), (5, 6)), ((1, 2), (7, 8)), ((3, 4), (5, 6)), ((3, 4), (7, 8))]
        self.assertEqual(list(x), ref)
        self.assertEqual(x.dimen, 2)
    finally:
        SetModule.FLATTEN_CROSS_PRODUCT = origFlattenCross
    self.assertIn(((1, 2), (5, 6)), x)
    self.assertIn((1, (2, 5), 6), x)
    self.assertIn((1, 2, 5, 6), x)
    self.assertNotIn((5, 6, 1, 2), x)