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
def build_tokens_line(self):
    """Build a logical line from tokens."""
    logical = []
    comments = []
    length = 0
    prev_row = prev_col = mapping = None
    for token_type, text, start, end, line in self.tokens:
        if token_type in SKIP_TOKENS:
            continue
        if not mapping:
            mapping = [(0, start)]
        if token_type == tokenize.COMMENT:
            comments.append(text)
            continue
        if token_type == tokenize.STRING:
            text = mute_string(text)
        if prev_row:
            start_row, start_col = start
            if prev_row != start_row:
                prev_text = self.lines[prev_row - 1][prev_col - 1]
                if prev_text == ',' or (prev_text not in '{[(' and text not in '}])'):
                    text = ' ' + text
            elif prev_col != start_col:
                text = line[prev_col:start_col] + text
        logical.append(text)
        length += len(text)
        mapping.append((length, end))
        prev_row, prev_col = end
    self.logical_line = ''.join(logical)
    self.noqa = comments and noqa(''.join(comments))
    return mapping