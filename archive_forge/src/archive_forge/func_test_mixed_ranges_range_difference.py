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
def test_mixed_ranges_range_difference(self):
    i = RangeSet(0, 10, 2)
    j = SetOf([0, 1, 2, 'a'])
    k = Any
    ir, = list(i.ranges())
    jr0, jr1, jr2, jr3 = list(j.ranges())
    kr, = list(k.ranges())
    self.assertEqual(ir.range_difference(i.ranges()), [])
    self.assertEqual(ir.range_difference([jr0]), [NR(2, 10, 2)])
    self.assertEqual(ir.range_difference([jr1]), [NR(0, 10, 2)])
    self.assertEqual(ir.range_difference([jr2]), [NR(0, 0, 0), NR(4, 10, 2)])
    self.assertEqual(ir.range_difference([jr3]), [NR(0, 10, 2)])
    self.assertEqual(ir.range_difference(j.ranges()), [NR(4, 10, 2)])
    self.assertEqual(ir.range_difference(k.ranges()), [])
    self.assertEqual(jr0.range_difference(i.ranges()), [])
    self.assertEqual(jr0.range_difference([jr0]), [])
    self.assertEqual(jr0.range_difference([jr1]), [jr0])
    self.assertEqual(jr0.range_difference([jr2]), [jr0])
    self.assertEqual(jr0.range_difference([jr3]), [jr0])
    self.assertEqual(jr0.range_difference(j.ranges()), [])
    self.assertEqual(jr0.range_difference(k.ranges()), [])
    self.assertEqual(jr1.range_difference(i.ranges()), [jr1])
    self.assertEqual(jr1.range_difference([jr0]), [jr1])
    self.assertEqual(jr1.range_difference([jr1]), [])
    self.assertEqual(jr1.range_difference([jr2]), [jr1])
    self.assertEqual(jr1.range_difference([jr3]), [jr1])
    self.assertEqual(jr1.range_difference(j.ranges()), [])
    self.assertEqual(jr1.range_difference(k.ranges()), [])
    self.assertEqual(jr3.range_difference(i.ranges()), [jr3])
    self.assertEqual(jr3.range_difference([jr0]), [jr3])
    self.assertEqual(jr3.range_difference([jr1]), [jr3])
    self.assertEqual(jr3.range_difference([jr2]), [jr3])
    self.assertEqual(jr3.range_difference([jr3]), [])
    self.assertEqual(jr3.range_difference(j.ranges()), [])
    self.assertEqual(jr3.range_difference(k.ranges()), [])
    self.assertEqual(kr.range_difference(i.ranges()), [kr])
    self.assertEqual(kr.range_difference([jr0]), [kr])
    self.assertEqual(kr.range_difference([jr1]), [kr])
    self.assertEqual(kr.range_difference([jr2]), [kr])
    self.assertEqual(kr.range_difference([jr3]), [kr])
    self.assertEqual(kr.range_difference(j.ranges()), [kr])
    self.assertEqual(kr.range_difference(k.ranges()), [])