import sys
import datetime
import locale as _locale
from itertools import repeat
def monthdays2calendar(self, year, month):
    """
        Return a matrix representing a month's calendar.
        Each row represents a week; week entries are
        (day number, weekday number) tuples. Day numbers outside this month
        are zero.
        """
    days = list(self.itermonthdays2(year, month))
    return [days[i:i + 7] for i in range(0, len(days), 7)]