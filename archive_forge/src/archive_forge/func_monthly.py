import datetime
import re
@staticmethod
def monthly(t):
    if t.month == 12:
        y, m = (t.year + 1, 1)
    else:
        y, m = (t.year, t.month + 1)
    return t.replace(year=y, month=m, day=1, hour=0, minute=0, second=0, microsecond=0)