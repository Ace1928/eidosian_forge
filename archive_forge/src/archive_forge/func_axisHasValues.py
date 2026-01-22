import copy
import os
import pickle
import warnings
import numpy as np
def axisHasValues(self, axis):
    ax = self._interpretAxis(axis)
    return 'values' in self._info[ax]