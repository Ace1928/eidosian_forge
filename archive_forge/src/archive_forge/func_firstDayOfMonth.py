import re, time, datetime
from .utils import isStr
def firstDayOfMonth(self):
    """returns (cloned) first day of month"""
    return self.__class__(self.__repr__()[-8:-2] + '01')