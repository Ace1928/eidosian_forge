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
def test_invalid_operators(self):
    m = ConcreteModel()
    m.I = RangeSet(5)
    m.J = Set([1, 2])
    with self.assertRaisesRegex(TypeError, 'Cannot apply a Set operator to an indexed Set component \\(J\\)'):
        m.I | m.J
    with self.assertRaisesRegex(TypeError, 'Cannot apply a Set operator to an indexed Set component \\(J\\)'):
        m.J | m.I
    m.x = Suffix()
    with self.assertRaisesRegex(TypeError, 'Cannot apply a Set operator to a non-Set Suffix component \\(x\\)'):
        m.I | m.x
    with self.assertRaisesRegex(TypeError, 'Cannot apply a Set operator to a non-Set Suffix component \\(x\\)'):
        m.x | m.I
    m.y = Var([1, 2])
    with self.assertRaisesRegex(TypeError, 'Cannot apply a Set operator to an indexed Var component \\(y\\)'):
        m.I | m.y
    with self.assertRaisesRegex(TypeError, 'Cannot apply a Set operator to a non-Set component data \\(y\\[1\\]\\)'):
        m.I | m.y[1]
    with self.assertRaisesRegex(TypeError, 'Cannot apply a Set operator to an indexed Var component \\(y\\)'):
        m.y | m.I
    with self.assertRaisesRegex(TypeError, 'Cannot apply a Set operator to a non-Set component data \\(y\\[1\\]\\)'):
        m.y[1] | m.I