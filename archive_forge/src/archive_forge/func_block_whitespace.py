from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import collections
import contextlib
import itertools
import tokenize
from six import StringIO
from pasta.base import formatting as fmt
from pasta.base import fstring_utils
def block_whitespace(self, indent_level):
    """Parses whitespace from the current _loc to the end of the block."""
    start_i = self._i
    full_whitespace = self.whitespace(comment=True)
    if not indent_level:
        return full_whitespace
    self._i = start_i
    lines = full_whitespace.splitlines(True)
    try:
        last_line_idx = next((i for i, line in reversed(list(enumerate(lines))) if line.startswith(indent_level + '#')))
    except StopIteration:
        self._loc = self._tokens[self._i].end
        return ''
    lines = lines[:last_line_idx + 1]
    end_line = self._tokens[self._i].end[0] + 1 + len(lines)
    list(self.takewhile(lambda tok: tok.start[0] < end_line))
    self._loc = self._tokens[self._i].end
    return ''.join(lines)