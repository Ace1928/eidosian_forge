from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
def dict_id(self):
    """Obtain the integer ID of the dictionary."""
    return int(lib.ZDICT_getDictID(self._data, len(self._data)))