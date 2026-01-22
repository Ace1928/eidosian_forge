import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def setResultsName(self, name, listAllMatches=False):
    ret = super(OneOrMore, self).setResultsName(name, listAllMatches)
    ret.saveAsList = True
    return ret