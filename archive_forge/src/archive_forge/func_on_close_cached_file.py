from __future__ import annotations
import os
import pickle
import time
from typing import TYPE_CHECKING
from fsspec.utils import atomic_write
def on_close_cached_file(self, f: Any, path: str) -> None:
    """Perform side-effect actions on closing a cached file.

        The actual closing of the file is the responsibility of the caller.
        """
    c = self.cached_files[-1][path]
    if c['blocks'] is not True and len(c['blocks']) * f.blocksize >= f.size:
        c['blocks'] = True