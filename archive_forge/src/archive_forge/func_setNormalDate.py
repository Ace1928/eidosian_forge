import re, time, datetime
from .utils import isStr
def setNormalDate(self, normalDate):
    NormalDate.setNormalDate(self, normalDate)
    self._checkDOW()