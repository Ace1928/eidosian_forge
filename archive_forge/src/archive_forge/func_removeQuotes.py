import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def removeQuotes(s, l, t):
    """Helper parse action for removing quotation marks from parsed quoted strings.
       To use, add this parse action to quoted string using::
         quotedString.setParseAction( removeQuotes )
    """
    return t[0][1:-1]