from pyomo.common.deprecation import deprecated
from pyomo.common.modeling import NOTSET
import pyomo.core.expr as EXPR
from pyomo.core.kernel.base import ICategorizedObject, _abstract_readwrite_property
from pyomo.core.kernel.container_utils import define_simple_containers
from pyomo.core.expr.numvalue import (
@deprecated('noclone() is deprecated and can be omitted: Pyomo expressions natively support shared subexpressions.', version='6.6.2')
def noclone(expr):
    try:
        if expr.is_potentially_variable():
            return expression(expr)
    except AttributeError:
        pass
    try:
        if is_constant(expr):
            return expr
    except:
        return expr
    return data_expression(expr)