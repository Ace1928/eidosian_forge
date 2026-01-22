import csv
import datetime
import os
def unsupported(self, date=None, result='codename'):
    """Get list of all unsupported distributions based on the given date."""
    if date is None:
        date = self._date
    supported = self.supported(date)
    distros = [self._format(result, x) for x in self._avail(date) if x.series not in supported]
    return distros