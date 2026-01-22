import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def extractText(s, l, t):
    del t[:]
    t.insert(0, s[t._original_start:t._original_end])
    del t['_original_start']
    del t['_original_end']