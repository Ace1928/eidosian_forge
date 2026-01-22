from __future__ import absolute_import
import cython
import os
import platform
from unicodedata import normalize
from contextlib import contextmanager
from .. import Utils
from ..Plex.Scanners import Scanner
from ..Plex.Errors import UnrecognizedInput
from .Errors import error, warning, hold_errors, release_errors, CompileError
from .Lexicon import any_string_prefix, make_lexicon, IDENT
from .Future import print_function
def expect_newline(self, message='Expected a newline', ignore_semicolon=False):
    useless_trailing_semicolon = None
    if ignore_semicolon and self.sy == ';':
        useless_trailing_semicolon = self.position()
        self.next()
    if self.sy != 'EOF':
        self.expect('NEWLINE', message)
    if useless_trailing_semicolon is not None:
        warning(useless_trailing_semicolon, 'useless trailing semicolon')