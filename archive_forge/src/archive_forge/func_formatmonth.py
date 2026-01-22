import sys
import datetime
import locale as _locale
from itertools import repeat
def formatmonth(self, theyear, themonth, withyear=True):
    """
        Return a formatted month as a table.
        """
    v = []
    a = v.append
    a('<table border="0" cellpadding="0" cellspacing="0" class="%s">' % self.cssclass_month)
    a('\n')
    a(self.formatmonthname(theyear, themonth, withyear=withyear))
    a('\n')
    a(self.formatweekheader())
    a('\n')
    for week in self.monthdays2calendar(theyear, themonth):
        a(self.formatweek(week))
        a('\n')
    a('</table>')
    a('\n')
    return ''.join(v)