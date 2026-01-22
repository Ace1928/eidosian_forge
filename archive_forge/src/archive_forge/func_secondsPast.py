import functools
import operator
import math
import datetime as DT
from matplotlib import _api
from matplotlib.dates import date2num
def secondsPast(self, frame, jd):
    t = self
    if frame != self._frame:
        t = self.convert(frame)
    delta = t._jd - jd
    return t._seconds + delta * 86400