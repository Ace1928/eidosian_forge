import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def setOffset(self, i):
    self.tup = (self.tup[0], i)