import sys
import functools
import difflib
import pprint
import re
import warnings
import collections
import contextlib
import traceback
import types
from . import result
from .util import (strclass, safe_repr, _count_diff_all_purpose,
@classmethod
def doClassCleanups(cls):
    """Execute all class cleanup functions. Normally called for you after
        tearDownClass."""
    cls.tearDown_exceptions = []
    while cls._class_cleanups:
        function, args, kwargs = cls._class_cleanups.pop()
        try:
            function(*args, **kwargs)
        except Exception:
            cls.tearDown_exceptions.append(sys.exc_info())