import re, time, datetime
from .utils import isStr
def formatUSCentury(self):
    """return date as string in 4-digit year US format: MM/DD/YYYY"""
    d = self.__repr__()
    return '%s/%s/%s' % (d[-4:-2], d[-2:], d[-8:-4])