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
def test_value_list_attr(self):
    m = ConcreteModel()
    m.J = Set(ordered=True, initialize=[1, 3, 2])
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.core'):
        tmp = m.J.value_list
    self.assertIs(type(tmp), list)
    self.assertEqual(tmp, list([1, 3, 2]))
    self.assertRegex(output.getvalue().replace('\n', ' '), "^DEPRECATED: The 'value_list' attribute is deprecated.  Use .ordered_data\\(\\)")