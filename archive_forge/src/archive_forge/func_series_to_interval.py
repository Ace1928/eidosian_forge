from collections.abc import MutableMapping
from pyomo.contrib.mpc.data.dynamic_data_base import _is_iterable, _DynamicDataBase
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.interval_data import IntervalData
from pyomo.contrib.mpc.data.find_nearest_index import find_nearest_interval_index
def series_to_interval(data, use_left_endpoints=False):
    """
    Arguments
    ---------
    data: TimeSeriesData
        Data that will be converted into an IntervalData object
    use_left_endpoints: Bool (optional)
        Flag indicating whether values on intervals should come
        from the values at the left or right endpoints of the
        intervals

    Returns
    -------
    IntervalData

    """
    time = data.get_time_points()
    data_dict = data.get_data()
    n_t = len(time)
    if n_t == 1:
        t0 = time[0]
        return IntervalData(data_dict, [(t0, t0)])
    else:
        new_data = {}
        intervals = [(time[i - 1], time[i]) for i in range(1, n_t)]
        for key, values in data_dict.items():
            interval_values = [values[i - 1] if use_left_endpoints else values[i] for i in range(1, n_t)]
            new_data[key] = interval_values
        return IntervalData(new_data, intervals)