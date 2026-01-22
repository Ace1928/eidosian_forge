import copy
import os
import pickle
import warnings
import numpy as np
def axisValues(self, axis):
    """Return the list of values for an axis"""
    ax = self._interpretAxis(axis)
    if 'values' in self._info[ax]:
        return self._info[ax]['values']
    else:
        raise Exception('Array axis %s (%d) has no associated values.' % (str(axis), ax))