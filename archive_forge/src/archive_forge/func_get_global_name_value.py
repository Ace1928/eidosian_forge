import math
from collections import OrderedDict
import numpy
from .base import numeric_types, string_types
from . import ndarray
from . import registry
def get_global_name_value(self):
    """Returns zipped name and value pairs for global results.

        Returns
        -------
        list of tuples
            A (name, value) tuple list.
        """
    if self._has_global_stats:
        name, value = self.get_global()
        if not isinstance(name, list):
            name = [name]
        if not isinstance(value, list):
            value = [value]
        return list(zip(name, value))
    else:
        return self.get_name_value()