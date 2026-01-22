import sys
import datetime
import locale as _locale
from itertools import repeat
def formatstring(cols, colwidth=_colwidth, spacing=_spacing):
    """Returns a string formatted from n strings, centered within n columns."""
    spacing *= ' '
    return spacing.join((c.center(colwidth) for c in cols))