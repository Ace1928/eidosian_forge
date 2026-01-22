import re
import warnings
import weakref
from collections import OrderedDict
from .. import functions as fn
from ..Qt import QtCore
from .ParameterItem import ParameterItem
def incrementName(self, name):
    base, num = re.match('([^\\d]*)(\\d*)', name).groups()
    numLen = len(num)
    if numLen == 0:
        num = 2
        numLen = 1
    else:
        num = int(num)
    while True:
        newName = base + '%%0%dd' % numLen % num
        if newName not in self.names:
            return newName
        num += 1