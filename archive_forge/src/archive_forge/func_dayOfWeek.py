import re, time, datetime
from .utils import isStr
def dayOfWeek(self):
    """return integer representing day of week, Mon=0, Tue=1, etc."""
    return dayOfWeek(*self.toTuple())