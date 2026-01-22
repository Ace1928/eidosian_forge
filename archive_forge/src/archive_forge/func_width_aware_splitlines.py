import re
from cwcwidth import wcswidth, wcwidth
from itertools import chain
from typing import (
from .escseqparse import parse, remove_ansi
from .termformatconstants import (
def width_aware_splitlines(self, columns: int) -> Iterator['FmtStr']:
    """Split into lines, pushing doublewidth characters at the end of a line to the next line.

        When a double-width character is pushed to the next line, a space is added to pad out the line.
        """
    if columns < 2:
        raise ValueError('Column width %s is too narrow.' % columns)
    if wcswidth(self.s, None) == -1:
        raise ValueError('bad values for width aware slicing')
    return self._width_aware_splitlines(columns)