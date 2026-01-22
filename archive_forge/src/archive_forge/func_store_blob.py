import atexit
import os
import shutil
import tempfile
import weakref
from fastimport.reftracker import RefTracker
from ... import lru_cache, trace
from . import branch_mapper
from .helpers import single_plural
def store_blob(self, id, data):
    """Store a blob of data."""
    if not self._blob_ref_counts or id in self._blob_ref_counts:
        self._sticky_blobs[id] = data
        self._sticky_memory_bytes += len(data)
        if self._sticky_memory_bytes > self._sticky_cache_size:
            self._flush_blobs_to_disk()
    elif data == b'':
        self._sticky_blobs[id] = data
    else:
        self._blobs[id] = data