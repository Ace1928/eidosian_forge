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
def next_of_type(self, token_type):
    """Parse a token of the given type and return it."""
    token = self.next()
    if token.type != token_type:
        raise ValueError('Expected %r but found %r\nline %d: %s' % (tokenize.tok_name[token_type], token.src, token.start[0], self.lines[token.start[0] - 1]))
    return token