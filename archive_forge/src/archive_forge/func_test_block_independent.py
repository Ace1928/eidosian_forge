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
def test_block_independent(self):
    m = ConcreteModel()
    with self.assertRaisesRegex(RuntimeError, "Cannot assign a GlobalSet 'Reals' to model 'unknown'"):
        m.a_set = Reals
    self.assertEqual(str(Reals), 'Reals')
    self.assertIsNone(Reals._parent)
    m.blk = Block()
    with self.assertRaisesRegex(RuntimeError, "Cannot assign a GlobalSet 'Reals' to block 'blk'"):
        m.blk.a_set = Reals
    self.assertEqual(str(Reals), 'Reals')
    self.assertIsNone(Reals._parent)