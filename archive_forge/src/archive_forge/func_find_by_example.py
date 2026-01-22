import sqlite3
import datetime
import warnings
from sqlalchemy import create_engine, Column, ForeignKey, Sequence
from sqlalchemy.engine.url import URL
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.interfaces import PoolListener
from sqlalchemy.orm import sessionmaker, deferred
from sqlalchemy.orm.exc import NoResultFound, MultipleResultsFound
from sqlalchemy.types import Integer, BigInteger, Boolean, DateTime, String, \
from sqlalchemy.sql.expression import asc, desc
from crash import Crash, Marshaller, pickle, HIGHEST_PROTOCOL
from textio import CrashDump
import win32
@Transactional
def find_by_example(self, crash, offset=None, limit=None):
    """
        Find all crash dumps that have common properties with the crash dump
        provided.

        Results can be paged to avoid consuming too much memory if the database
        is large.

        @see: L{find}

        @type  crash: L{Crash}
        @param crash: Crash object to compare with. Fields set to C{None} are
            ignored, all other fields but the signature are used in the
            comparison.

            To search for signature instead use the L{find} method.

        @type  offset: int
        @param offset: (Optional) Skip the first I{offset} results.

        @type  limit: int
        @param limit: (Optional) Return at most I{limit} results.

        @rtype:  list(L{Crash})
        @return: List of similar crash dumps found.
        """
    if limit is not None and (not limit):
        warnings.warn('CrashDAO.find_by_example() was set a limit of 0 results, returning without executing a query.')
        return []
    query = self._session.query(CrashDTO)
    query = query.asc(CrashDTO.id)
    dto = CrashDTO(crash)
    for name, column in compat.iteritems(CrashDTO.__dict__):
        if not name.startswith('__') and name not in ('id', 'signature', 'data'):
            if isinstance(column, Column):
                value = getattr(dto, name, None)
                if value is not None:
                    query = query.filter(column == value)
    if offset:
        query = query.offset(offset)
    if limit:
        query = query.limit(limit)
    try:
        return [dto.toCrash() for dto in query.all()]
    except NoResultFound:
        return []