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
def dots(self, num_dots):
    """Parse a number of dots.
    
    This is to work around an oddity in python3's tokenizer, which treats three
    `.` tokens next to each other in a FromImport's level as an ellipsis. This
    parses until the expected number of dots have been seen.
    """
    result = ''
    dots_seen = 0
    prev_loc = self._loc
    while dots_seen < num_dots:
        tok = self.next()
        assert tok.src in ('.', '...')
        result += self._space_between(prev_loc, tok.start) + tok.src
        dots_seen += tok.src.count('.')
        prev_loc = self._loc
    return result