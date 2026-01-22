from statsmodels.compat.python import lmap, lrange
from itertools import cycle, zip_longest
import csv
def get_aligns(self, output_format, **fmt_dict):
    """Return string, sequence of column alignments.
        Ensure comformable data_aligns in `fmt_dict`."""
    fmt = self._get_fmt(output_format, **fmt_dict)
    return ''.join((cell.alignment(output_format, **fmt) for cell in self))