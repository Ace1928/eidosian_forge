from pyomo.common.deprecation import deprecated
from pyomo.common.modeling import NOTSET
import pyomo.core.expr as EXPR
from pyomo.core.kernel.base import ICategorizedObject, _abstract_readwrite_property
from pyomo.core.kernel.container_utils import define_simple_containers
from pyomo.core.expr.numvalue import (
def is_parameter_type(self):
    """A boolean indicating whether this expression is a parameter object."""
    return False