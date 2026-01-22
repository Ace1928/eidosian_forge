from pyomo.contrib.mpc.data.dynamic_data_base import _is_iterable
from pyomo.contrib.mpc.data.find_nearest_index import (
def load_data_from_series(data, model, time, tolerance=0.0):
    """A function to load TimeSeriesData into a model

    Arguments
    ---------
    data: TimeSeriesData
    model: BlockData
    time: Iterable

    """
    time_list = list(time)
    time_indices = [find_nearest_index(time_list, t, tolerance=tolerance) for t in data.get_time_points()]
    for idx, t in zip(time_indices, data.get_time_points()):
        if idx is None:
            raise RuntimeError('Time point %s not found time set' % t)
    if len(time_list) != len(data.get_time_points()):
        raise RuntimeError('TimeSeriesData object and model must have same number of time points to load data from series')
    data = data.get_data()
    for cuid, vals in data.items():
        var = model.find_component(cuid)
        if var is None:
            _raise_invalid_cuid(cuid, model)
        for idx, val in zip(time_indices, vals):
            t = time_list[idx]
            var[t].set_value(val)