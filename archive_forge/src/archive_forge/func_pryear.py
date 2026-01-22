import sys
import datetime
import locale as _locale
from itertools import repeat
def pryear(self, theyear, w=0, l=0, c=6, m=3):
    """Print a year's calendar."""
    print(self.formatyear(theyear, w, l, c, m), end='')