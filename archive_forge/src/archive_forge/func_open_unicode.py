from __future__ import absolute_import, unicode_literals
import io
import posixpath
import sys
from os import environ
from pybtex.exceptions import PybtexError
from pybtex.kpathsea import kpsewhich
def open_unicode(filename, mode='r', encoding=None):
    if encoding is None:
        encoding = get_default_encoding()
    return _open(io.open, filename, mode, encoding=encoding)