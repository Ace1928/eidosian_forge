from __future__ import annotations
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Union
import sqlalchemy
from langchain_core._api import deprecated
from langchain_core.utils import get_from_env
from sqlalchemy import (
from sqlalchemy.engine import Engine, Result
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError
from sqlalchemy.schema import CreateTable
from sqlalchemy.sql.expression import Executable
from sqlalchemy.types import NullType
@classmethod
def from_uri(cls, database_uri: str, engine_args: Optional[dict]=None, **kwargs: Any) -> SQLDatabase:
    """Construct a SQLAlchemy engine from URI."""
    _engine_args = engine_args or {}
    return cls(create_engine(database_uri, **_engine_args), **kwargs)