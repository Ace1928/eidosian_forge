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
def enable_left_recursion(cache_size_limit: typing.Optional[int]=None, *, force=False) -> None:
    """
        Enables "bounded recursion" parsing, which allows for both direct and indirect
        left-recursion. During parsing, left-recursive :class:`Forward` elements are
        repeatedly matched with a fixed recursion depth that is gradually increased
        until finding the longest match.

        Example::

            from pip._vendor import pyparsing as pp
            pp.ParserElement.enable_left_recursion()

            E = pp.Forward("E")
            num = pp.Word(pp.nums)
            # match `num`, or `num '+' num`, or `num '+' num '+' num`, ...
            E <<= E + '+' - num | num

            print(E.parse_string("1+2+3"))

        Recursion search naturally memoizes matches of ``Forward`` elements and may
        thus skip reevaluation of parse actions during backtracking. This may break
        programs with parse actions which rely on strict ordering of side-effects.

        Parameters:

        - ``cache_size_limit`` - (default=``None``) - memoize at most this many
          ``Forward`` elements during matching; if ``None`` (the default),
          memoize all ``Forward`` elements.

        Bounded Recursion parsing works similar but not identical to Packrat parsing,
        thus the two cannot be used together. Use ``force=True`` to disable any
        previous, conflicting settings.
        """
    if force:
        ParserElement.disable_memoization()
    elif ParserElement._packratEnabled:
        raise RuntimeError('Packrat and Bounded Recursion are not compatible')
    if cache_size_limit is None:
        ParserElement.recursion_memos = _UnboundedMemo()
    elif cache_size_limit > 0:
        ParserElement.recursion_memos = _LRUMemo(capacity=cache_size_limit)
    else:
        raise NotImplementedError('Memo size of %s' % cache_size_limit)
    ParserElement._left_recursion_enabled = True