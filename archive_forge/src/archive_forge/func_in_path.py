import binascii
import os
import posixpath
import stat
import warnings
import zlib
from collections import namedtuple
from hashlib import sha1
from io import BytesIO
from typing import (
from .errors import (
from .file import GitFile
def in_path(self, path: bytes):
    """Return a copy of this entry with the given path prepended."""
    if not isinstance(self.path, bytes):
        raise TypeError('Expected bytes for path, got %r' % path)
    return TreeEntry(posixpath.join(path, self.path), self.mode, self.sha)