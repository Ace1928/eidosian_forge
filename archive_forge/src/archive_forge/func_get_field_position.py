from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
def get_field_position(self, field):
    try:
        return self._fields.index(field)
    except ValueError:
        raise KeyError('%s does not have a field named %r' % (self.__class__.__name__, field))