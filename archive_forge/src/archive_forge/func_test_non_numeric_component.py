import subprocess
import sys
from math import nan, inf
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy, numpy_available
from pyomo.environ import (
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.expr.numvalue import (
from pyomo.common.numeric_types import _native_boolean_types
def test_non_numeric_component(self):
    m = ConcreteModel()
    m.v = Var([1, 2])
    with self.assertRaisesRegex(TypeError, "The 'IndexedVar' object 'v' is not a valid type for Pyomo numeric expressions"):
        as_numeric(m.v)
    obj = PyomoObject()
    with self.assertRaisesRegex(TypeError, "The 'PyomoObject' object '.*' is not a valid type for Pyomo numeric expressions"):
        as_numeric(obj)