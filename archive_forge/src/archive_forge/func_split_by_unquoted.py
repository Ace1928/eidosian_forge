import sys
import string
import fileinput
import re
import os
import copy
import platform
import codecs
from pathlib import Path
from . import __version__
from .auxfuncs import *
from . import symbolic
def split_by_unquoted(line, characters):
    """
    Splits the line into (line[:i], line[i:]),
    where i is the index of first occurrence of one of the characters
    not within quotes, or len(line) if no such index exists
    """
    assert not set('"\'') & set(characters), 'cannot split by unquoted quotes'
    r = re.compile('\\A(?P<before>({single_quoted}|{double_quoted}|{not_quoted})*)(?P<after>{char}.*)\\Z'.format(not_quoted='[^"\'{}]'.format(re.escape(characters)), char='[{}]'.format(re.escape(characters)), single_quoted="('([^'\\\\]|(\\\\.))*')", double_quoted='("([^"\\\\]|(\\\\.))*")'))
    m = r.match(line)
    if m:
        d = m.groupdict()
        return (d['before'], d['after'])
    return (line, '')