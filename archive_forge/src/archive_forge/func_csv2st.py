from statsmodels.compat.python import lmap, lrange
from itertools import cycle, zip_longest
import csv
def csv2st(csvfile, headers=False, stubs=False, title=None):
    """Return SimpleTable instance,
    created from the data in `csvfile`,
    which is in comma separated values format.
    The first row may contain headers: set headers=True.
    The first column may contain stubs: set stubs=True.
    Can also supply headers and stubs as tuples of strings.
    """
    rows = list()
    with open(csvfile, encoding='utf-8') as fh:
        reader = csv.reader(fh)
        if headers is True:
            headers = next(reader)
        elif headers is False:
            headers = ()
        if stubs is True:
            stubs = list()
            for row in reader:
                if row:
                    stubs.append(row[0])
                    rows.append(row[1:])
        else:
            for row in reader:
                if row:
                    rows.append(row)
        if stubs is False:
            stubs = ()
    ncols = len(rows[0])
    if any((len(row) != ncols for row in rows)):
        raise OSError('All rows of CSV file must have same length.')
    return SimpleTable(data=rows, headers=headers, stubs=stubs)