import re, time, datetime
from .utils import isStr
def month(self):
    """returns month as integer 1-12"""
    return int(repr(self.normalDate)[-4:-2])