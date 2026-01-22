import sys
import datetime
import locale as _locale
from itertools import repeat
def formatweek(self, theweek):
    """
        Return a complete week as a table row.
        """
    s = ''.join((self.formatday(d, wd) for d, wd in theweek))
    return '<tr>%s</tr>' % s