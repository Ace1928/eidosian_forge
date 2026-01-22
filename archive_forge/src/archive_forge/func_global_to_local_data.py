from collections import OrderedDict
import importlib
def global_to_local_data(self, global_data):
    if type(global_data) is list:
        local_data = list()
        assert len(global_data) == self._n_total_tasks
        for i in self._local_map:
            local_data.append(global_data[i])
        return local_data
    elif type(global_data) is OrderedDict:
        local_data = OrderedDict()
        assert len(global_data) == self._n_total_tasks
        idx = 0
        for k, v in global_data.items():
            if idx in self._local_map:
                local_data[k] = v
            idx += idx
        return local_data
    raise ValueError('Unknown type passed to global_to_local_data. Expected list or OrderedDict.')