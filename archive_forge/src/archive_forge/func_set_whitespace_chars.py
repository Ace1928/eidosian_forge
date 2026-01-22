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
def set_whitespace_chars(self, chars: Union[Set[str], str], copy_defaults: bool=False) -> 'ParserElement':
    """
        Overrides the default whitespace chars
        """
    self.skipWhitespace = True
    self.whiteChars = set(chars)
    self.copyDefaultWhiteChars = copy_defaults
    return self