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
@classmethod
def get_by_owner(cls, owner_id: int, owner_key: str='owner_id', name: Optional[str]=None, name_key: Optional[str]='name', skip: int=0, limit: int=100, order_by: Optional[Any]=None, load_attrs: Optional[List[str]]=None, load_attr_method: Optional[Union[str, Callable]]=None, readonly: Optional[bool]=False):
    """
        Fetch all records by owner_id
        """
    with PostgresDB.session(ro=readonly) as db_sess:
        query = select(cls).where(getattr(cls, owner_key) == owner_id)
        if name:
            query = query.where(getattr(cls, name_key) == name)
        if load_attrs:
            load_attr_method = get_attr_func(load_attr_method)
            for attr in load_attrs:
                query = query.options(load_attr_method(getattr(cls, attr)))
        if skip is not None:
            query = query.offset(skip)
        if limit:
            query = query.limit(limit)
        if order_by is not None:
            query = query.order_by(order_by)
        results = db_sess.execute(query)
        return results.scalars().all()