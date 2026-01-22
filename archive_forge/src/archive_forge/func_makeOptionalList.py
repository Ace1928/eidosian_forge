import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def makeOptionalList(n):
    if n > 1:
        return Optional(self + makeOptionalList(n - 1))
    else:
        return Optional(self)