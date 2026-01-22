from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.base.expression import Expression
from pyomo.core.base.set import Set
from pyomo.contrib.mpc.data.series_data import get_indexed_cuid
from pyomo.contrib.mpc.data.scalar_data import ScalarData
def get_penalty_at_time(variables, t, target_data, weight_data=None, time_set=None, variable_set=None):
    """Returns an Expression penalizing the deviation of the specified
    variables at the specified point in time from the specified target

    Arguments
    ---------
    variables: List
        List of time-indexed variables that will be penalized
    t: Float
        Time point at which to apply the penalty
    target_data: ScalarData
        ScalarData object containing the target for (at least) the variables
        to be penalized
    weight_data: ScalarData (optional)
        ScalarData object containing the penalty weights for (at least) the
        variables to be penalized
    time_set: Set (optional)
        Time set that indexes the provided variables. This is only used if
        target or weight data are provided as a ComponentMap with VarData
        as keys. In this case the Set is necessary to recover the CUIDs
        used internally as keys
    variable_set: Set (optional)
        Set indexing the list of variables provided, if such a set already
        exists

    Returns
    -------
    Set, Expression
        Set indexing the list of variables provided and an Expression,
        indexed by this set, containing the weighted penalty expressions

    """
    if variable_set is None:
        variable_set = Set(initialize=range(len(variables)))
    penalty_expressions = _get_penalty_expressions_at_time(variables, t, target_data, weight_data=weight_data, time_set=time_set)

    def penalty_rule(m, i):
        return penalty_expressions[i]
    penalty = Expression(variable_set, rule=penalty_rule)
    return (variable_set, penalty)