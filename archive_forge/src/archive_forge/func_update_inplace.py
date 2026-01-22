import datetime
import contextlib
from sqlalchemy import func
from sqlalchemy import delete as sqlalchemy_delete
from sqlalchemy import update as sqlalchemy_update
from sqlalchemy import exists as sqlalchemy_exists
from sqlalchemy import insert as sqlalchemy_insert
from sqlalchemy.future import select
from sqlalchemy.sql.expression import Select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import selectinload, joinedload, immediateload
from sqlalchemy import Column, Integer, DateTime, String, Text, ForeignKey, Boolean, Identity, Enum
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from typing import Any, Generator, AsyncGenerator, Iterable, Optional, Union, Type, Dict, cast, TYPE_CHECKING, List, Tuple, TypeVar, Callable
from lazyops.utils import create_unique_id, create_timestamp
from lazyops.utils.logs import logger
from lazyops.types import lazyproperty
from lazyops.libs.psqldb.base import Base, PostgresDB, AsyncSession, Session
from lazyops.libs.psqldb.utils import SQLJson, get_pydantic_model, object_serializer
from fastapi.exceptions import HTTPException
from pydantic import BaseModel
def update_inplace(self, session: Optional[Session]=None, _raise_exceptions: Optional[bool]=True, _verbose: Optional[bool]=False, **kwargs) -> SyncSQLModelT:
    """
        Update a record in place
        """
    with self.session(session=session) as db_sess:
        for key, value in kwargs.items():
            setattr(self, key, value)
        try:
            local_object = db_sess.merge(self)
            db_sess.add(local_object)
            db_sess.commit()
            db_sess.flush()
            self = local_object
        except Exception as e:
            if _raise_exceptions:
                self._handle_exception(error=e, verbose=_verbose)
            elif _verbose:
                logger.trace(f'Error updating {self.__class__.__name__}', error=e)
        return self