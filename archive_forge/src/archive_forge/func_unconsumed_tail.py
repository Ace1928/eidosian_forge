from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
@property
def unconsumed_tail(self):
    """Data that has not yet been fed into the decompressor."""
    return b''