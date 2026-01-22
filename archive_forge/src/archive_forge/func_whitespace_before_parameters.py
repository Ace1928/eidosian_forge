from __future__ import with_statement
import inspect
import keyword
import os
import re
import sys
import time
import tokenize
import warnings
from fnmatch import fnmatch
from optparse import OptionParser
def whitespace_before_parameters(logical_line, tokens):
    """Avoid extraneous whitespace.

    Avoid extraneous whitespace in the following situations:
    - before the open parenthesis that starts the argument list of a
      function call.
    - before the open parenthesis that starts an indexing or slicing.

    Okay: spam(1)
    E211: spam (1)

    Okay: dict['key'] = list[index]
    E211: dict ['key'] = list[index]
    E211: dict['key'] = list [index]
    """
    prev_type, prev_text, __, prev_end, __ = tokens[0]
    for index in range(1, len(tokens)):
        token_type, text, start, end, __ = tokens[index]
        if token_type == tokenize.OP and text in '([' and (start != prev_end) and (prev_type == tokenize.NAME or prev_text in '}])') and (index < 2 or tokens[index - 2][1] != 'class') and (not keyword.iskeyword(prev_text)):
            yield (prev_end, "E211 whitespace before '%s'" % text)
        prev_type = token_type
        prev_text = text
        prev_end = end