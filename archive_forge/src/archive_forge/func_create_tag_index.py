from __future__ import annotations
import os
import io
import abc
import zlib
import errno
import time
import struct
import sqlite3
import threading
import pickletools
import asyncio
import inspect
import dill as pkl
import functools as ft
import contextlib as cl
import warnings
from fileio.lib.types import File, FileLike
from typing import Any, Optional, Type, Dict, Union, Tuple, TYPE_CHECKING
from lazyops.utils.pooler import ThreadPooler
from lazyops.libs.sqlcache.constants import (
from lazyops.libs.sqlcache.config import SqlCacheConfig
from lazyops.libs.sqlcache.exceptions import (
from lazyops.libs.sqlcache.utils import (
def create_tag_index(self):
    """Create tag index on cache database.
        It is better to initialize cache with `tag_index=True` than use this.
        :raises Timeout: if database timeout occurs
        """
    sql = self._sql
    sql(f'CREATE INDEX IF NOT EXISTS {self.table_name}_tag_rowid ON {self.table_name}(tag, rowid)')
    self.reset('tag_index', 1)