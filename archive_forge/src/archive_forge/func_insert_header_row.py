from statsmodels.compat.python import lmap, lrange
from itertools import cycle, zip_longest
import csv
def insert_header_row(self, rownum, headers, dec_below='header_dec_below'):
    """Return None.  Insert a row of headers,
        where ``headers`` is a sequence of strings.
        (The strings may contain newlines, to indicated multiline headers.)
        """
    header_rows = [header.split('\n') for header in headers]
    rows = list(zip_longest(*header_rows, **dict(fillvalue='')))
    rows.reverse()
    for i, row in enumerate(rows):
        self.insert(rownum, row, datatype='header')
        if i == 0:
            self[rownum].dec_below = dec_below
        else:
            self[rownum].dec_below = None