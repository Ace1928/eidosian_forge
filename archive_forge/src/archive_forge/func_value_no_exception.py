from pyomo.common.collections import ComponentSet
from pyomo.core.expr import identify_variables
from pyomo.environ import Constraint, value
def value_no_exception(c, div0=None):
    """
    Get value and ignore most exceptions (including division by 0).

    Args:
        c: a Pyomo component to get the value of
    Returns:
        A value, could be None
    """
    try:
        return value(c, exception=False)
    except ZeroDivisionError:
        return div0