import atexit
import os
import shutil
import tempfile
import weakref
from fastimport.reftracker import RefTracker
from ... import lru_cache, trace
from . import branch_mapper
from .helpers import single_plural
def fetch_blob(self, id):
    """Fetch a blob of data."""
    if id in self._blobs:
        return self._blobs.pop(id)
    if id in self._disk_blobs:
        offset, n_bytes, fn = self._disk_blobs[id]
        if fn is None:
            f = self._cleanup.small_blobs
            f.seek(offset)
            content = f.read(n_bytes)
        else:
            with open(fn, 'rb') as fp:
                content = fp.read()
        self._decref(id, self._disk_blobs, fn)
        return content
    content = self._sticky_blobs[id]
    if self._decref(id, self._sticky_blobs, None):
        self._sticky_memory_bytes -= len(content)
    return content