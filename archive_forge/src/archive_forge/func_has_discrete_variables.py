from pyomo.common import deprecated
from pyomo.core.base import Var
@deprecated('This function has been moved to `pyomo.util.blockutil`', version='5.6.9')
def has_discrete_variables(block):
    from pyomo.util.blockutil import has_discrete_variables
    return has_discrete_variables(block)