import re, time, datetime
from .utils import isStr
def year(self):
    """return year in yyyy format, negative values indicate B.C."""
    return int(repr(self.normalDate)[:-4])