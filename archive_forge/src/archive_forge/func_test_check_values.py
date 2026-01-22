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
def test_check_values(self):
    m = ConcreteModel()
    m.I = Set(ordered=True, initialize=[1, 3, 2])
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.core'):
        self.assertTrue(m.I.check_values())
    self.assertRegex(output.getvalue(), '^DEPRECATED: check_values\\(\\) is deprecated: Sets only contain valid')
    m.J = m.I * m.I
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.core', logging.DEBUG):
        self.assertTrue(m.J.check_values())
    self.assertRegex(output.getvalue(), '^DEPRECATED: check_values\\(\\) is deprecated:')
    m.K = Set([1, 2], ordered=True, initialize=[1, 3, 2])
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.core'):
        self.assertTrue(m.K.check_values())
    self.assertRegex(output.getvalue(), '^DEPRECATED: check_values\\(\\) is deprecated: Sets only contain valid')