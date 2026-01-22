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
def disable_memoization() -> None:
    """
        Disables active Packrat or Left Recursion parsing and their memoization

        This method also works if neither Packrat nor Left Recursion are enabled.
        This makes it safe to call before activating Packrat nor Left Recursion
        to clear any previous settings.
        """
    ParserElement.reset_cache()
    ParserElement._left_recursion_enabled = False
    ParserElement._packratEnabled = False
    ParserElement._parse = ParserElement._parseNoCache