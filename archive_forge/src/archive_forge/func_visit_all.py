from collections import deque
import os
import typing
from typing import (
from abc import ABC, abstractmethod
from enum import Enum
import string
import copy
import warnings
import re
import sys
from collections.abc import Iterable
import traceback
import types
from operator import itemgetter
from functools import wraps
from threading import RLock
from pathlib import Path
from .util import (
from .exceptions import *
from .actions import *
from .results import ParseResults, _ParseResultsWithOffset
from .unicode import pyparsing_unicode
def visit_all(self):
    """General-purpose method to yield all expressions and sub-expressions
        in a grammar. Typically just for internal use.
        """
    to_visit = deque([self])
    seen = set()
    while to_visit:
        cur = to_visit.popleft()
        if cur in seen:
            continue
        seen.add(cur)
        to_visit.extend(cur.recurse())
        yield cur