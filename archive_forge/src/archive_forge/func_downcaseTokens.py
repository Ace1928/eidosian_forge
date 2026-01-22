import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def downcaseTokens(s, l, t):
    """Helper parse action to convert tokens to lower case."""
    return [tt.lower() for tt in map(_ustr, t)]