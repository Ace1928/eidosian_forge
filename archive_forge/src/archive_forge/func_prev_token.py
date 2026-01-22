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
def prev_token(self, tok, include_extra=False):
    """
    Returns the previous token before the given one. If include_extra is True, includes non-coding
    tokens from the tokenize module, such as NL and COMMENT.
    """
    i = tok.index - 1
    if not include_extra:
        while is_non_coding_token(self._tokens[i].type):
            i -= 1
    return self._tokens[i]