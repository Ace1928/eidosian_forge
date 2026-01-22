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
def init_schemas(self, conn: 'sqlite3.Connection', exists: Optional[bool]=None) -> None:
    """
        Initializes the schemas
        """
    if exists is None:
        exists = self.sql_data_path.exists()
    if not self.sql_lock.is_locked:
        with self.sql_lock.acquire(10.0):
            index_schemas = self.get_index_schemas()
            for tablename, schema in index_schemas.items():
                if not isinstance(schema, list):
                    schema = [schema]
                for s in schema:
                    s.execute_sql_init(conn, tablename=tablename, skip_index=exists and (not self.force_reindex), auto_set=self.auto_set_schema)
    self._schemas_initialized = True