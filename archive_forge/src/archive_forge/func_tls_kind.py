from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@property
def tls_kind(self):
    """Return the thread-local storage (TLS) kind of this cursor."""
    if not hasattr(self, '_tls_kind'):
        self._tls_kind = conf.lib.clang_getCursorTLSKind(self)
    return TLSKind.from_id(self._tls_kind)