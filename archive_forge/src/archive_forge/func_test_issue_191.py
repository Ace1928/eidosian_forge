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
def test_issue_191(self):
    m = ConcreteModel()
    m.s = Set(['s1', 's2'], initialize=[1, 2, 3])
    m.s2 = Set(initialize=['a', 'b', 'c'])
    m.p = Param(m.s['s1'], initialize=10)
    temp = m.s['s1'] * m.s2
    m.v = Var(temp, initialize=5)
    self.assertEqual(len(m.v), 9)
    m.v_1 = Var(m.s['s1'], m.s2, initialize=10)
    self.assertEqual(len(m.v_1), 9)