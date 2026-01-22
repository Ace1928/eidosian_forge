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
def truncate_word(content: Any, *, length: int, suffix: str='...') -> str:
    """
    Truncate a string to a certain number of words, based on the max string
    length.
    """
    if not isinstance(content, str) or length <= 0:
        return content
    if len(content) <= length:
        return content
    return content[:length - len(suffix)].rsplit(' ', 1)[0] + suffix