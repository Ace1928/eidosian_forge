import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def setDebugActions(self, startAction, successAction, exceptionAction):
    """Enable display of debugging messages while doing pattern matching."""
    self.debugActions = (startAction or _defaultStartDebugAction, successAction or _defaultSuccessDebugAction, exceptionAction or _defaultExceptionDebugAction)
    self.debug = True
    return self