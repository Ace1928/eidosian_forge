import subprocess
import sys
from math import nan, inf
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy, numpy_available
from pyomo.environ import (
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.expr.numvalue import (
from pyomo.common.numeric_types import _native_boolean_types
def test_unknownNumericType(self):
    ref = MyBogusNumericType(42)
    self.assertNotIn(MyBogusNumericType, native_numeric_types)
    self.assertNotIn(MyBogusNumericType, native_types)
    try:
        val = as_numeric(ref)
        self.assertEqual(val().val, 42.0)
    finally:
        native_numeric_types.remove(MyBogusNumericType)
        native_types.remove(MyBogusNumericType)