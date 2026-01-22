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
def test_virtual(self):
    m = ConcreteModel()
    m.I = Set(initialize=[1, 2, 3])
    m.J = m.I * m.I
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.core', logging.DEBUG):
        self.assertFalse(m.I.virtual)
    self.assertRegex(output.getvalue(), "^DEPRECATED: The 'virtual' attribute is no longer supported")
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.core', logging.DEBUG):
        self.assertTrue(m.J.virtual)
    self.assertRegex(output.getvalue(), "^DEPRECATED: The 'virtual' attribute is no longer supported")
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.core'):
        m.J.virtual = True
    self.assertRegex(output.getvalue(), "^DEPRECATED: The 'virtual' attribute is no longer supported")
    with self.assertRaisesRegex(ValueError, "Attempting to set the \\(deprecated\\) 'virtual' attribute on J to an invalid value \\(False\\)"):
        m.J.virtual = False