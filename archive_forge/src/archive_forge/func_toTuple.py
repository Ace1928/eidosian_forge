import re, time, datetime
from .utils import isStr
def toTuple(self):
    """return date as (year, month, day) tuple"""
    return (self.year(), self.month(), self.day())