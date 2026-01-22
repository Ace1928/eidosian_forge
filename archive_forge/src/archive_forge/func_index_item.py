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
def index_item(self, item: 'SQLiteModelMixin', **kwargs) -> None:
    """
        Indexes the item
        """
    if self.enable_internal_index:
        if item.sql_tablename not in self.internal_index:
            self.internal_index[item.sql_tablename] = {}
        self.internal_index[item.sql_tablename][item.sql_pkey] = item