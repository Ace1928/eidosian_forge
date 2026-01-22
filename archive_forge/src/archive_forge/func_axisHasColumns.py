import copy
import os
import pickle
import warnings
import numpy as np
def axisHasColumns(self, axis):
    ax = self._interpretAxis(axis)
    return 'cols' in self._info[ax]