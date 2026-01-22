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
def test_issue_1375(self):

    def a_rule(m):
        for i in range(0):
            yield i

    def b_rule(m):
        for i in range(3):
            for j in range(0):
                yield (i, j)
    m = ConcreteModel()
    m.a = Set(initialize=a_rule, dimen=1)
    self.assertEqual(len(m.a), 0)
    m.b = Set(initialize=b_rule, dimen=2)
    self.assertEqual(len(m.b), 0)