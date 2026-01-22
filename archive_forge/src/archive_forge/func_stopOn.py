import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
import pprint
import traceback
import types
from datetime import datetime
from operator import itemgetter
import itertools
from functools import wraps
from contextlib import contextmanager
def stopOn(self, ender):
    if isinstance(ender, basestring):
        ender = self._literalStringClass(ender)
    self.not_ender = ~ender if ender is not None else None
    return self