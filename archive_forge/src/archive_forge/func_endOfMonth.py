import re, time, datetime
from .utils import isStr
def endOfMonth(self):
    """returns (cloned) last day of month"""
    return self.__class__(self.__repr__()[-8:-2] + str(self.lastDayOfMonth()))