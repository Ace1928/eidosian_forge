import numpy as np
from .casting import best_float, floor_exact, int_abs, shared_range, type_info
from .volumeutils import array_to_file, finite_range
def calc_scale(self, force=False):
    """Calculate / set scaling for floats/(u)ints to (u)ints"""
    if not force and self._scale_calced:
        return
    self.reset()
    if not self.scaling_needed():
        return
    self._do_scaling()
    self._scale_calced = True