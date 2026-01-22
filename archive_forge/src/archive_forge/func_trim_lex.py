from __future__ import absolute_import
import re
import sys
import os
import tokenize
from io import StringIO
from ._looper import looper
from .compat3 import bytes, unicode_, basestring_, next, is_unicode, coerce_text
def trim_lex(tokens):
    """
    Takes a lexed set of tokens, and removes whitespace when there is
    a directive on a line by itself:

       >>> tokens = lex('{{if x}}\\nx\\n{{endif}}\\ny', trim_whitespace=False)
       >>> tokens
       [('if x', (1, 3)), '\\nx\\n', ('endif', (3, 3)), '\\ny']
       >>> trim_lex(tokens)
       [('if x', (1, 3)), 'x\\n', ('endif', (3, 3)), 'y']
    """
    last_trim = None
    for i, current in enumerate(tokens):
        if isinstance(current, basestring_):
            continue
        item = current[0]
        if not statement_re.search(item) and item not in single_statements:
            continue
        if not i:
            prev = ''
        else:
            prev = tokens[i - 1]
        if i + 1 >= len(tokens):
            next_chunk = ''
        else:
            next_chunk = tokens[i + 1]
        if not isinstance(next_chunk, basestring_) or not isinstance(prev, basestring_):
            continue
        prev_ok = not prev or trail_whitespace_re.search(prev)
        if i == 1 and (not prev.strip()):
            prev_ok = True
        if last_trim is not None and last_trim + 2 == i and (not prev.strip()):
            prev_ok = 'last'
        if prev_ok and (not next_chunk or lead_whitespace_re.search(next_chunk) or (i == len(tokens) - 2 and (not next_chunk.strip()))):
            if prev:
                if i == 1 and (not prev.strip()) or prev_ok == 'last':
                    tokens[i - 1] = ''
                else:
                    m = trail_whitespace_re.search(prev)
                    prev = prev[:m.start() + 1]
                    tokens[i - 1] = prev
            if next_chunk:
                last_trim = i
                if i == len(tokens) - 2 and (not next_chunk.strip()):
                    tokens[i + 1] = ''
                else:
                    m = lead_whitespace_re.search(next_chunk)
                    next_chunk = next_chunk[m.end():]
                    tokens[i + 1] = next_chunk
    return tokens