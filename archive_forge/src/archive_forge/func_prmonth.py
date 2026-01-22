import sys
import datetime
import locale as _locale
from itertools import repeat
def prmonth(self, theyear, themonth, w=0, l=0):
    """
        Print a month's calendar.
        """
    print(self.formatmonth(theyear, themonth, w, l), end='')