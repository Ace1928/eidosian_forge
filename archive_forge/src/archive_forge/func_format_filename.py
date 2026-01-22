from __future__ import print_function, unicode_literals
import sys
import typing
from fs.path import abspath, join, normpath
def format_filename(fname):
    """Format a filename."""
    if not with_color:
        return fname
    if fname.startswith('.'):
        fname = '\x1b[33m%s\x1b[0m' % fname
    return fname