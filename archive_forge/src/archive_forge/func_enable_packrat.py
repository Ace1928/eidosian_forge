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
@staticmethod
def enable_packrat(cache_size_limit: int=128, *, force: bool=False) -> None:
    """
        Enables "packrat" parsing, which adds memoizing to the parsing logic.
        Repeated parse attempts at the same string location (which happens
        often in many complex grammars) can immediately return a cached value,
        instead of re-executing parsing/validating code.  Memoizing is done of
        both valid results and parsing exceptions.

        Parameters:

        - ``cache_size_limit`` - (default= ``128``) - if an integer value is provided
          will limit the size of the packrat cache; if None is passed, then
          the cache size will be unbounded; if 0 is passed, the cache will
          be effectively disabled.

        This speedup may break existing programs that use parse actions that
        have side-effects.  For this reason, packrat parsing is disabled when
        you first import pyparsing.  To activate the packrat feature, your
        program must call the class method :class:`ParserElement.enable_packrat`.
        For best results, call ``enable_packrat()`` immediately after
        importing pyparsing.

        Example::

            from pip._vendor import pyparsing
            pyparsing.ParserElement.enable_packrat()

        Packrat parsing works similar but not identical to Bounded Recursion parsing,
        thus the two cannot be used together. Use ``force=True`` to disable any
        previous, conflicting settings.
        """
    if force:
        ParserElement.disable_memoization()
    elif ParserElement._left_recursion_enabled:
        raise RuntimeError('Packrat and Bounded Recursion are not compatible')
    if not ParserElement._packratEnabled:
        ParserElement._packratEnabled = True
        if cache_size_limit is None:
            ParserElement.packrat_cache = _UnboundedCache()
        else:
            ParserElement.packrat_cache = _FifoCache(cache_size_limit)
        ParserElement._parse = ParserElement._parseCache