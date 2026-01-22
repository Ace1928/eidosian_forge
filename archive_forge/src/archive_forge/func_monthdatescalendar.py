import sys
import datetime
import locale as _locale
from itertools import repeat
def monthdatescalendar(self, year, month):
    """
        Return a matrix (list of lists) representing a month's calendar.
        Each row represents a week; week entries are datetime.date values.
        """
    dates = list(self.itermonthdates(year, month))
    return [dates[i:i + 7] for i in range(0, len(dates), 7)]