import contextlib
import decimal
import uuid
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Sequence, Union
from sqlalchemy import (
from sqlalchemy.ext.asyncio import (
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Query, Session, sessionmaker
from langchain.indexes.base import RecordManager
def list_keys(self, *, before: Optional[float]=None, after: Optional[float]=None, group_ids: Optional[Sequence[str]]=None, limit: Optional[int]=None) -> List[str]:
    """List records in the SQLite database based on the provided date range."""
    session: Session
    with self._make_session() as session:
        query: Query = session.query(UpsertionRecord).filter(UpsertionRecord.namespace == self.namespace)
        if after:
            query = query.filter(UpsertionRecord.updated_at > after)
        if before:
            query = query.filter(UpsertionRecord.updated_at < before)
        if group_ids:
            query = query.filter(UpsertionRecord.group_id.in_(group_ids))
        if limit:
            query = query.limit(limit)
        records = query.all()
    return [r.key for r in records]