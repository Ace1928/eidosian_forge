from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
from warnings import warn
from passlib import exc, registry
from passlib.context import CryptContext
from passlib.exc import ExpectedStringError
from passlib.hash import htdigest
from passlib.utils import render_bytes, to_bytes, is_ascii_codec
from passlib.utils.decor import deprecated_method
from passlib.utils.compat import join_bytes, unicode, BytesIO, PY3
def load_if_changed(self):
    """Reload from ``self.path`` only if file has changed since last load"""
    if not self._path:
        raise RuntimeError('%r is not bound to a local file' % self)
    if self._mtime and self._mtime == os.path.getmtime(self._path):
        return False
    self.load()
    return True