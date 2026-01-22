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
def fixValue(self):
    """
        changes lower bound and upper bound to the initial value if exists.
        :return: None
        """
    val = self.varValue
    if val is not None:
        self.bounds(val, val)