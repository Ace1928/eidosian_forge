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
def isViolated(self):
    """
        returns true if the penalty variables are non-zero
        """
    if abs(value(self.denominator)) >= const.EPS:
        if self.lowTarget is not None:
            if self.lowTarget > self.findLHSValue():
                return True
        if self.upTarget is not None:
            if self.findLHSValue() > self.upTarget:
                return True
    else:
        return False