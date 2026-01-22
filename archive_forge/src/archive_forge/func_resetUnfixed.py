import copy
from collections import OrderedDict
from math import log2
import numpy as np
from .. import functions as fn
def resetUnfixed(self):
    """
        For any variable that does not have a fixed value, reset
        its value to None.
        """
    for var in self._vars.values():
        if var[2] != 'fixed':
            var[0] = None