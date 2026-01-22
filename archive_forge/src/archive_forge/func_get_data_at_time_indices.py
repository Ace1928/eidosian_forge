from collections import namedtuple
from pyomo.core.expr.numvalue import value as pyo_value
from pyomo.contrib.mpc.data.find_nearest_index import find_nearest_index
from pyomo.contrib.mpc.data.get_cuid import get_indexed_cuid
from pyomo.contrib.mpc.data.dynamic_data_base import _is_iterable, _DynamicDataBase
from pyomo.contrib.mpc.data.scalar_data import ScalarData
def get_data_at_time_indices(self, indices):
    """
        Returns data at the specified index or indices of this object's list
        of time points.

        """
    if _is_iterable(indices):
        index_list = list(sorted(indices))
        time_list = [self._time[i] for i in indices]
        data = {cuid: [values[idx] for idx in index_list] for cuid, values in self._data.items()}
        time_set = self._orig_time_set
        return TimeSeriesData(data, time_list, time_set=time_set)
    else:
        return ScalarData({cuid: values[indices] for cuid, values in self._data.items()})