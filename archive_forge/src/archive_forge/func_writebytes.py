from __future__ import absolute_import, print_function, unicode_literals
import typing
from six import text_type
from . import errors
from .base import FS
from .memoryfs import MemoryFS
from .mode import validate_open_mode, validate_openbin_mode
from .path import abspath, forcedir, normpath
def writebytes(self, path, contents):
    self.check()
    fs, _path = self._delegate(path)
    return fs.writebytes(_path, contents)