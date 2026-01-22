from itertools import zip_longest
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
import pyomo.kernel as pmo
from pyomo.util.components import iter_component, rename_components
def con_rule(m, i):
    return m.x[i] + m.z == i