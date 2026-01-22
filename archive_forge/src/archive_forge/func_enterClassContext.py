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
def enterClassContext(cls, cm):
    """Same as enterContext, but class-wide."""
    return _enter_context(cm, cls.addClassCleanup)