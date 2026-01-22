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
def whitespace_around_keywords(logical_line):
    """Avoid extraneous whitespace around keywords.

    Okay: True and False
    E271: True and  False
    E272: True  and False
    E273: True and\\tFalse
    E274: True\\tand False
    """
    for match in KEYWORD_REGEX.finditer(logical_line):
        before, after = match.groups()
        if '\t' in before:
            yield (match.start(1), 'E274 tab before keyword')
        elif len(before) > 1:
            yield (match.start(1), 'E272 multiple spaces before keyword')
        if '\t' in after:
            yield (match.start(2), 'E273 tab after keyword')
        elif len(after) > 1:
            yield (match.start(2), 'E271 multiple spaces after keyword')