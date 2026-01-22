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
def doCleanups(self):
    """Execute all cleanup functions. Normally called for you after
        tearDown."""
    outcome = self._outcome or _Outcome()
    while self._cleanups:
        function, args, kwargs = self._cleanups.pop()
        with outcome.testPartExecutor(self):
            self._callCleanup(function, *args, **kwargs)
    return outcome.success