from __future__ import with_statement
import inspect
import keyword
import os
import re
import sys
import time
import tokenize
import warnings
from fnmatch import fnmatch
from optparse import OptionParser
def trailing_blank_lines(physical_line, lines, line_number, total_lines):
    """Trailing blank lines are superfluous.

    Okay: spam(1)
    W391: spam(1)\\n

    However the last line should end with a new line (warning W292).
    """
    if line_number == total_lines:
        stripped_last_line = physical_line.rstrip()
        if not stripped_last_line:
            return (0, 'W391 blank line at end of file')
        if stripped_last_line == physical_line:
            return (len(physical_line), 'W292 no newline at end of file')