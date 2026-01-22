from pyomo.common.collections import ComponentSet
from pyomo.core.expr import identify_variables
from pyomo.environ import Constraint, value
def count_free_variables(blk):
    """
    Count free variables that are in active equality constraints.  Ignore
    inequalities, because this is used in the degrees of freedom calculations
    """
    return len(free_variables_in_active_equalities_set(blk))