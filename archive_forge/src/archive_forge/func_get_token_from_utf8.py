import abc
import ast
import bisect
import sys
import token
from ast import Module
from typing import Iterable, Iterator, List, Optional, Tuple, Any, cast, TYPE_CHECKING
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
from .line_numbers import LineNumbers
from .util import (
def get_token_from_utf8(self, lineno, col_offset):
    """
    Same as get_token(), but interprets col_offset as a UTF8 offset, which is what `ast` uses.
    """
    return self.get_token(lineno, self._line_numbers.from_utf8_col(lineno, col_offset))