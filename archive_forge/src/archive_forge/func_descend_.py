import sys
import pprint as _pprint_
from pyomo.common.collections import ComponentMap
import pyomo.core
from pyomo.core.expr.numvalue import NumericValue
from pyomo.core.kernel.base import (
def descend_(obj_):
    if obj_._is_heterogeneous_container:
        return False
    else:
        return descend(obj_)