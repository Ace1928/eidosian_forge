import sys
import datetime
import locale as _locale
from itertools import repeat
def formatweekheader(self):
    """
        Return a header for a week as a table row.
        """
    s = ''.join((self.formatweekday(i) for i in self.iterweekdays()))
    return '<tr>%s</tr>' % s