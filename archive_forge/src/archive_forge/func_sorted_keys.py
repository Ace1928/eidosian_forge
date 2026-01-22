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
def sorted_keys(self):
    """
        returns the list of keys sorted by name
        """
    result = [(v.name, v) for v in self.keys()]
    result.sort()
    result = [v for _, v in result]
    return result