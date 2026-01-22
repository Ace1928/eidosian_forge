import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def charsAsStr(s):
    if len(s) > 4:
        return s[:4] + '...'
    else:
        return s