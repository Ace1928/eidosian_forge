from collections.abc import Sized
import logging
from pyomo.core.kernel.block import block
from pyomo.core.kernel.set_types import IntegerSet
from pyomo.core.kernel.variable import variable, variable_dict, variable_tuple
from pyomo.core.kernel.constraint import (
from pyomo.core.kernel.expression import expression, expression_tuple
import pyomo.core.kernel.piecewise_library.util
@property
def triangulation(self):
    """The triangulation over the domain of this function"""
    return self._f.triangulation