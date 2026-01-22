from __future__ import annotations
import os
import pickle
import time
from typing import TYPE_CHECKING
from fsspec.utils import atomic_write
def pop_file(self, path: str) -> str | None:
    """Remove metadata of cached file.

        If path is in the cache, return the filename of the cached file,
        otherwise return ``None``.  Caller is responsible for deleting the
        cached file.
        """
    details = self.check_file(path, None)
    if not details:
        return None
    _, fn = details
    if fn.startswith(self._storage[-1]):
        self.cached_files[-1].pop(path)
        self.save()
    else:
        raise PermissionError('Can only delete cached file in last, writable cache location')
    return fn