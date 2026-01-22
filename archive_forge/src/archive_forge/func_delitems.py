import array
import logging
import posixpath
import warnings
from collections.abc import MutableMapping
from functools import cached_property
from fsspec.core import url_to_fs
def delitems(self, keys):
    """Remove multiple keys from the store"""
    self.fs.rm([self._key_to_str(k) for k in keys])