import asyncio
import collections
import contextvars
import datetime as dt
import inspect
import functools
import numbers
import os
import re
import sys
import traceback
import warnings
from collections import defaultdict, OrderedDict
from contextlib import contextmanager
from numbers import Real
from textwrap import dedent
from threading import get_ident
from collections import abc
@_deprecated()
def is_ordered_dict(d):
    """
    Predicate checking for ordered dictionaries. OrderedDict is always
    ordered, and vanilla Python dictionaries are ordered for Python 3.6+

    .. deprecated:: 2.0.0
    """
    py3_ordered_dicts = sys.version_info.major == 3 and sys.version_info.minor >= 6
    vanilla_odicts = sys.version_info.major > 3 or py3_ordered_dicts
    return isinstance(d, OrderedDict) or (vanilla_odicts and isinstance(d, dict))