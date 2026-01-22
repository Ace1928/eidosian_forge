from __future__ import annotations
import abc
import atexit
import sqlite3
import pathlib
import filelock
import contextlib
from pydantic import BaseModel, Field, model_validator, model_serializer
from lazyops.imports._aiosqlite import resolve_aiosqlite
from lazyops.utils.lazy import lazy_import
from lazyops.utils import logger, Timer
from typing import Optional, List, Dict, Any, Union, Type, Tuple, TypeVar, AsyncGenerator, overload, TYPE_CHECKING
@property
def sql_lock(self) -> filelock.SoftFileLock:
    """
        Returns the SQL Lock
        """
    if self._sql_lock is None:
        self._sql_lock = filelock.SoftFileLock(self.sql_lock_path.as_posix(), timeout=0, thread_local=False)
        if self.ephemeral:
            atexit.register(self.cleanup_on_exit)
        atexit.register(self.save_on_exit)
    return self._sql_lock