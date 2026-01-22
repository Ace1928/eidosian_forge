import re, time, datetime
from .utils import isStr
def setMonth(self, month):
    """set the month [1-12]"""
    if month < 1 or month > 12:
        raise NormalDateException('month is outside range 1 to 12')
    y, m, d = self.toTuple()
    self.setNormalDate((y, month, d))