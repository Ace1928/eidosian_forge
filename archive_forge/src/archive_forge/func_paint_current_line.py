import logging
import itertools
from curtsies import fsarray, fmtstr, FSArray
from curtsies.formatstring import linesplit
from curtsies.fmtfuncs import bold
from .parse import func_for_letter
def paint_current_line(rows, columns, current_display_line):
    lines = display_linize(current_display_line, columns, True)
    return fsarray(lines, width=columns)