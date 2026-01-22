from functools import cached_property
from numba.core import ir, analysis, transforms, ir_utils
def get_yield_points(self):
    """
        Return an iterable of YieldPoint instances.
        """
    return self.yield_points.values()