import re
import sys
from docutils import DataError
from docutils.utils import strip_combining_chars
def init_row(self, colspec, offset):
    i = 0
    cells = []
    for start, end in colspec:
        morecols = 0
        try:
            assert start == self.columns[i][0]
            while end != self.columns[i][1]:
                i += 1
                morecols += 1
        except (AssertionError, IndexError):
            raise TableMarkupError('Column span alignment problem in table line %s.' % (offset + 2), offset=offset + 1)
        cells.append([0, morecols, offset, []])
        i += 1
    return cells