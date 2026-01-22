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
@classmethod
def using_each(cls, seq, **class_kwargs):
    """
        Yields a sequence of class(obj, **class_kwargs) for obj in seq.

        Example::

            LPAR, RPAR, LBRACE, RBRACE, SEMI = Suppress.using_each("(){};")

        """
    yield from (cls(obj, **class_kwargs) for obj in seq)