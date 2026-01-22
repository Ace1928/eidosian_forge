import contextlib
from sqlalchemy import func, text
from sqlalchemy import delete as sqlalchemy_delete
from sqlalchemy import update as sqlalchemy_update
from sqlalchemy import exists as sqlalchemy_exists
from sqlalchemy.future import select
from sqlalchemy.sql.expression import Select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import selectinload, joinedload, immediateload
from sqlalchemy import Column, Integer, DateTime, String, Text, ForeignKey, Boolean, Identity, Enum
from typing import Any, Generator, AsyncGenerator, Iterable, Optional, Union, Type, Dict, cast, TYPE_CHECKING, List, Tuple, TypeVar, Callable
from lazyops.utils import create_unique_id, create_timestamp
from lazyops.utils.logs import logger
from lazyops.types import lazyproperty
from lazyops.libs.psqldb.base import Base, PostgresDB, AsyncSession, Session
from lazyops.libs.psqldb.utils import SQLJson, get_pydantic_model, object_serializer, get_sqlmodel_dict
from fastapi.exceptions import HTTPException
from pydantic import BaseModel
def sqlmodel_dict(self, exclude: Optional[List[str]]=None, include: Optional[List[str]]=None, exclude_none: Optional[bool]=False, **kwargs) -> Dict[str, Any]:
    """
        Return a dictionary representation of the model.
        """
    model_dict = get_sqlmodel_dict(self)
    if exclude is not None:
        model_dict = {key: value for key, value in model_dict.items() if key not in exclude}
    if include is not None:
        model_dict = {key: value for key, value in model_dict.items() if key in include}
    if exclude_none:
        model_dict = {key: value for key, value in model_dict.items() if value is not None}
    return model_dict