from pyomo.contrib.mpc.data.dynamic_data_base import _is_iterable
from pyomo.contrib.mpc.data.find_nearest_index import (
A function to load IntervalData into a model

    Loads values into specified variables at time points that are
    within the intervals specified. If a time point is on the boundary
    of two intervals, the default is to use the interval on the left.
    Often, intervals should be treated as half-open, i.e. one of the
    left or right endpoints should be excluded. This can be enforced
    with the corresponding optional arguments.

    Arguments
    ---------
    data: IntervalData
    model: BlockData
    time: Iterable
    tolerance: Float
    prefer_left: Bool
    exclude_left_endpoint: Bool
    exclude_right_endpoint: Bool

    