from functools import cached_property
from numba.core import ir, analysis, transforms, ir_utils
@cached_property
def usedefs(self):
    return analysis.compute_use_defs(self._blocks)