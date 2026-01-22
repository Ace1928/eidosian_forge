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
def init_conn(self, timeout: int, **kwargs):
    """
        Handles initialization of connection
        """
    sql = self._sql_retry
    try:
        current_settings = dict(sql(f'SELECT key, value FROM Settings_{self.table_name}').fetchall())
    except sqlite3.OperationalError:
        current_settings = {}
    sets = self.settings.copy()
    sets.update(current_settings)
    for key in METADATA:
        sets.pop(key, None)
    for key, value in sorted(sets.items()):
        if key.startswith('sqlite_'):
            self.reset(key, value, update=False)
    sql(f'CREATE TABLE IF NOT EXISTS Settings_{self.table_name} ( key TEXT NOT NULL UNIQUE, value)')
    kwargs = {key[5:]: value for key, value in sets.items() if key.startswith('disk_')}
    self._medium = self._medium_type(self.connection_path, **kwargs)
    for key, value in sets.items():
        query = f'INSERT OR REPLACE INTO Settings_{self.table_name} VALUES (?, ?)'
        sql(query, (key, value))
        self.reset(key, value)
    for key, value in METADATA.items():
        query = f'INSERT OR IGNORE INTO Settings_{self.table_name} VALUES (?, ?)'
        sql(query, (key, value))
        self.reset(key)
    (self._page_size,), = sql('PRAGMA page_size').fetchall()
    sql(f'CREATE TABLE IF NOT EXISTS {self.table_name} ( rowid INTEGER PRIMARY KEY, key BLOB, raw INTEGER, store_time REAL, expire_time REAL, access_time REAL, access_count INTEGER DEFAULT 0, tag BLOB, size INTEGER DEFAULT 0, mode INTEGER DEFAULT 0, filename TEXT, value BLOB)')
    sql(f'CREATE UNIQUE INDEX IF NOT EXISTS Store_key_raw ON {self.table_name}(key, raw)')
    sql(f'CREATE INDEX IF NOT EXISTS Store_expire_time ON {self.table_name} (expire_time)')
    query = self.eviction_policy['init']
    if query is not None:
        sql(query)
    sql(f'CREATE TRIGGER IF NOT EXISTS Settings_{self.table_name}_count_insert AFTER INSERT ON {self.table_name} FOR EACH ROW BEGIN UPDATE Settings_{self.table_name} SET value = value + 1 WHERE key = "count"; END')
    sql(f'CREATE TRIGGER IF NOT EXISTS Settings_{self.table_name}_count_delete AFTER DELETE ON {self.table_name} FOR EACH ROW BEGIN UPDATE Settings_{self.table_name} SET value = value - 1 WHERE key = "count"; END')
    sql(f'CREATE TRIGGER IF NOT EXISTS Settings_{self.table_name}_size_insert AFTER INSERT ON {self.table_name} FOR EACH ROW BEGIN UPDATE Settings_{self.table_name} SET value = value + NEW.size WHERE key = "size"; END')
    sql(f'CREATE TRIGGER IF NOT EXISTS Settings_{self.table_name}_size_update AFTER UPDATE ON {self.table_name} FOR EACH ROW BEGIN UPDATE Settings_{self.table_name} SET value = value + NEW.size - OLD.size WHERE key = "size"; END')
    sql(f'CREATE TRIGGER IF NOT EXISTS Settings_{self.table_name}_size_delete AFTER DELETE ON {self.table_name} FOR EACH ROW BEGIN UPDATE Settings_{self.table_name} SET value = value - OLD.size WHERE key = "size"; END')
    if self.config.tag_index:
        self.create_tag_index()
    else:
        self.drop_tag_index()
    self.close()
    self._timeout = timeout
    self._sql