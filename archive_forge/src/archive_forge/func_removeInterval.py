from __future__ import annotations
import errno
import os
import sys
import warnings
from typing import AnyStr
from collections import OrderedDict
from typing import (
from incremental import Version
from twisted.python.deprecate import deprecatedModuleAttribute
def removeInterval(self, interval):
    for i in range(len(self.intervals)):
        if self.intervals[i][1] == interval:
            index = self.intervals[i][2]
            del self.intervals[i]
            for i in self.intervals:
                if i[2] > index:
                    i[2] -= 1
            return
    raise ValueError('Specified interval not in IntervalDifferential')