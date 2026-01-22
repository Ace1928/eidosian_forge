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
def comparison_negative(logical_line):
    """Negative comparison should be done using "not in" and "is not".

    Okay: if x not in y:\\n    pass
    Okay: assert (X in Y or X is Z)
    Okay: if not (X in Y):\\n    pass
    Okay: zz = x is not y
    E713: Z = not X in Y
    E713: if not X.B in Y:\\n    pass
    E714: if not X is Y:\\n    pass
    E714: Z = not X.B is Y
    """
    match = COMPARE_NEGATIVE_REGEX.search(logical_line)
    if match:
        pos = match.start(1)
        if match.group(2) == 'in':
            yield (pos, "E713 test for membership should be 'not in'")
        else:
            yield (pos, "E714 test for object identity should be 'is not'")