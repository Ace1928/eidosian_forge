from __future__ import annotations
import gc
import abc
import asyncio
import datetime
import contextlib
from pathlib import Path
from pydantic.networks import PostgresDsn
from pydantic_settings import BaseSettings
from pydantic import validator, model_validator, computed_field, BaseModel, Field, PrivateAttr
from sqlalchemy import text as sql_text, TextClause
from sqlalchemy.pool import NullPool, AsyncAdaptedQueuePool
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine
from lazyops.utils.logs import logger, Logger
from lazyops.utils.lazy import lazy_import
from ...utils.helpers import update_dict
from typing import Any, Dict, List, Optional, Type, Literal, Iterable, Tuple, TypeVar, Union, Annotated, Callable, Generator, AsyncGenerator, Set, TYPE_CHECKING
def get_engine_kwargs(self, readonly: Optional[bool]=False, verbose: Optional[bool]=False, **engine_kwargs) -> Dict[str, Any]:
    """
        Get the Engine KWargs
        """
    kwargs = self.engine_kwargs or {}
    if engine_kwargs:
        kwargs = update_dict(kwargs, engine_kwargs)
    kwargs = update_dict(kwargs, self.engine_ro_kwargs) if readonly else update_dict(kwargs, self.engine_rw_kwargs)
    self.log_readonly_warning(verbose) if readonly else self.log_db_url(verbose)
    kwargs['url'] = str(self.readonly_url or self.url if readonly else self.url)
    if self.engine_json_serializer:
        kwargs['json_serializer'] = self.engine_json_serializer
    if self.engine_poolclass:
        kwargs['poolclass'] = self.engine_poolclass
    return kwargs