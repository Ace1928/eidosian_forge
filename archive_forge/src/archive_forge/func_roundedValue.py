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
def roundedValue(self, eps=1e-05):
    if self.cat == const.LpInteger and self.varValue != None and (abs(self.varValue - round(self.varValue)) <= eps):
        return round(self.varValue)
    else:
        return self.varValue