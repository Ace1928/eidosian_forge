from pyomo.core.base.set import Set
from pyomo.contrib.mpc.data.get_cuid import get_indexed_cuid
def get_data_from_key(self, key, context=None):
    """
        Returns the value associated with the given key.

        """
    cuid = get_indexed_cuid(key, (self._orig_time_set,), context=context)
    return self._data[cuid]