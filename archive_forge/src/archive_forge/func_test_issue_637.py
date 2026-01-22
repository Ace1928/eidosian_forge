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
def test_issue_637(self):
    constraints = {c for c in itertools.product(['constrA', 'constrB'], range(5))}
    vars = {v for v in itertools.product(['var1', 'var2', 'var3'], range(5))}
    matrix_coefficients = {m for m in itertools.product(constraints, vars)}
    m = ConcreteModel()
    m.IDX = Set(initialize=matrix_coefficients)
    m.Matrix = Param(m.IDX, default=0)
    self.assertEqual(len(m.Matrix), 2 * 5 * 3 * 5)