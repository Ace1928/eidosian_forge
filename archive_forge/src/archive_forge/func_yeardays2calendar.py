import sys
import datetime
import locale as _locale
from itertools import repeat
def yeardays2calendar(self, year, width=3):
    """
        Return the data for the specified year ready for formatting (similar to
        yeardatescalendar()). Entries in the week lists are
        (day number, weekday number) tuples. Day numbers outside this month are
        zero.
        """
    months = [self.monthdays2calendar(year, i) for i in range(January, January + 12)]
    return [months[i:i + width] for i in range(0, len(months), width)]