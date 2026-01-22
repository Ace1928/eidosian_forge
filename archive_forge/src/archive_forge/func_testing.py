import csv
import datetime
import os
def testing(self, date=None, result='codename'):
    """Get latest testing Debian distribution based on the given date."""
    if date is None:
        date = self._date
    distros = [x for x in self._avail(date) if x.release is None and x.version or (x.release is not None and date < x.release and (x.eol is None or date <= x.eol))]
    if not distros:
        raise DistroDataOutdated()
    return self._format(result, distros[-1])