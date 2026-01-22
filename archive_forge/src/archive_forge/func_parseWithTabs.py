import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def parseWithTabs(self):
    """Overrides default behavior to expand C{<TAB>}s to spaces before parsing the input string.
           Must be called before C{parseString} when the input grammar contains elements that
           match C{<TAB>} characters."""
    self.keepTabs = True
    return self