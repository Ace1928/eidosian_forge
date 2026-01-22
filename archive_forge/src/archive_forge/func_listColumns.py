import copy
import os
import pickle
import warnings
import numpy as np
def listColumns(self, axis=None):
    """Return a list of column names for axis. If axis is not specified, then return a dict of {axisName: (column names), ...}."""
    if axis is None:
        ret = {}
        for i in range(self.ndim):
            if 'cols' in self._info[i]:
                cols = [c['name'] for c in self._info[i]['cols']]
            else:
                cols = []
            ret[self.axisName(i)] = cols
        return ret
    else:
        axis = self._interpretAxis(axis)
        return [c['name'] for c in self._info[axis]['cols']]