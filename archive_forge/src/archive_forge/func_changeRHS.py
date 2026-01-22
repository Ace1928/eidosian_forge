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
def changeRHS(self, RHS):
    """
        alters the RHS of a constraint so that it can be modified in a resolve
        """
    self.constant = -RHS
    self.modified = True