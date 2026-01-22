import numpy as np
from .casting import best_float, floor_exact, int_abs, shared_range, type_info
from .volumeutils import array_to_file, finite_range
@property
def has_nan(self):
    """True if array has NaNs"""
    if self._has_nan is None:
        if self._array.dtype.kind in 'fc':
            self.finite_range()
        else:
            self._has_nan = False
    return self._has_nan