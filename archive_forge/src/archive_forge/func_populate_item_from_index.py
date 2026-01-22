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
def populate_item_from_index(self, tablename: str, pkey: Union[str, int, Dict[str, Any], List[Union[str, int]]], **kwargs) -> Optional['SQLiteModelMixin']:
    """
        Populates the item from the index
        """
    if tablename not in self.internal_index:
        return None
    if isinstance(pkey, dict):
        pkey = list(pkey.values())
    if not isinstance(pkey, list):
        pkey = [pkey]
    return next((self.internal_index[tablename][pk] for pk in pkey if pk is not None and pk in self.internal_index[tablename]), None)