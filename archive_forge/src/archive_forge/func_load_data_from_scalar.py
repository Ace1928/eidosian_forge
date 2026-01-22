from pyomo.contrib.mpc.data.dynamic_data_base import _is_iterable
from pyomo.contrib.mpc.data.find_nearest_index import (
def load_data_from_scalar(data, model, time):
    """A function to load ScalarData into a model

    Arguments
    ---------
    data: ScalarData
    model: BlockData
    time: Iterable

    """
    data = data.get_data()
    t_iter = time if _is_iterable(time) else (time,)
    for cuid, val in data.items():
        var = model.find_component(cuid)
        if var is None:
            _raise_invalid_cuid(cuid, model)
        if var.is_indexed():
            for t in t_iter:
                var[t].set_value(val)
        else:
            var.set_value(val)