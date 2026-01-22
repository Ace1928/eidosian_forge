import warnings
import functools
from .. import symbol, init, ndarray
from ..base import string_types, numeric_types
def pack_weights(self, args):
    return _cells_pack_weights(self._cells, args)