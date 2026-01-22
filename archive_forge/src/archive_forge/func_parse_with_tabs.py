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
def parse_with_tabs(self) -> 'ParserElement':
    """
        Overrides default behavior to expand ``<TAB>`` s to spaces before parsing the input string.
        Must be called before ``parse_string`` when the input grammar contains elements that
        match ``<TAB>`` characters.
        """
    self.keepTabs = True
    return self