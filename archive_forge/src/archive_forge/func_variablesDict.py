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
def variablesDict(self):
    variables = {}
    if self.objective:
        for v in self.objective:
            variables[v.name] = v
    for c in list(self.constraints.values()):
        for v in c:
            variables[v.name] = v
    return variables