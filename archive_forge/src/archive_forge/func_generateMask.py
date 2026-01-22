from collections import OrderedDict
import numpy as np
from .. import functions as fn
from .. import parametertree as ptree
from ..Qt import QtCore
def generateMask(self, data, startMask):
    vals = data[self.fieldName][startMask]
    mask = np.ones(len(vals), dtype=bool)
    otherMask = np.ones(len(vals), dtype=bool)
    for c in self:
        key = c.maskValue
        if key == '__other__':
            m = ~otherMask
        else:
            m = vals != key
            otherMask &= m
        if c.value() is False:
            mask &= m
    startMask[startMask] = mask
    return startMask