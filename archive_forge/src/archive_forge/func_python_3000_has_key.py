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
def python_3000_has_key(logical_line, noqa):
    """The {}.has_key() method is removed in Python 3: use the 'in' operator.

    Okay: if "alph" in d:\\n    print(d["alph"])
    W601: assert d.has_key('alph')
    """
    pos = logical_line.find('.has_key(')
    if pos > -1 and (not noqa):
        yield (pos, "W601 .has_key() is deprecated, use 'in'")