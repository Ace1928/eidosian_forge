from pyomo.common.collections import ComponentMap
from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.base.expression import Expression
from pyomo.core.base.set import Set
from pyomo.contrib.mpc.data.series_data import get_indexed_cuid
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.interval_data import IntervalData
from pyomo.contrib.mpc.data.convert import interval_to_series, _process_to_dynamic_data
def get_penalty_from_piecewise_constant_target(variables, time, setpoint_data, weight_data=None, variable_set=None, tolerance=0.0, prefer_left=True):
    """Returns an IndexedExpression penalizing deviation between
    the specified variables and piecewise constant target data.

    Arguments
    ---------
    variables: List of Pyomo variables
        Variables that participate in the cost expressions.
    time: Iterable
        Index used for the cost expression
    setpoint_data: IntervalData
        Holds the piecewise constant values that will be used as
        setpoints
    weight_data: ScalarData (optional)
        Weights for variables. Default is all ones.
    tolerance: Float (optional)
        Tolerance used for determining whether a time point
        is within an interval. Default is zero.
    prefer_left: Bool (optional)
        If a time point lies at the boundary of two intervals, whether
        the value on the left will be chosen. Default is True.

    Returns
    -------
    Set, Expression
        Pyomo Expression, indexed by time, for the total weighted
        tracking cost with respect to the provided setpoint.

    """
    if variable_set is None:
        variable_set = Set(initialize=range(len(variables)))
    if isinstance(setpoint_data, IntervalData):
        setpoint_time_series = interval_to_series(setpoint_data, time_points=time, tolerance=tolerance, prefer_left=prefer_left)
    else:
        setpoint_time_series = IntervalData(*setpoint_data)
    var_set, tracking_cost = get_penalty_from_time_varying_target(variables, time, setpoint_time_series, weight_data=weight_data, variable_set=variable_set)
    return (var_set, tracking_cost)