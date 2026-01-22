from collections import Counter
import sys
import warnings
from time import time
from .apis import LpSolverDefault, PULP_CBC_CMD
from .apis.core import clock
from .utilities import value
from . import constants as const
from . import mps_lp as mpslp
import logging
import re
def infeasibilityGap(self, mip=1):
    gap = 0
    for v in self.variables():
        gap = max(abs(v.infeasibilityGap(mip)), gap)
    for c in self.constraints.values():
        if not c.valid(0):
            gap = max(abs(c.value()), gap)
    return gap