from __future__ import annotations
import io
import logging
import os
import threading
import warnings
import weakref
from errno import ESPIPE
from glob import has_magic
from hashlib import sha256
from typing import ClassVar
from .callbacks import DEFAULT_CALLBACK
from .config import apply_config, conf
from .dircache import DirCache
from .transaction import Transaction
from .utils import (
@classmethod
def clear_instance_cache(cls):
    """
        Clear the cache of filesystem instances.

        Notes
        -----
        Unless overridden by setting the ``cachable`` class attribute to False,
        the filesystem class stores a reference to newly created instances. This
        prevents Python's normal rules around garbage collection from working,
        since the instances refcount will not drop to zero until
        ``clear_instance_cache`` is called.
        """
    cls._cache.clear()