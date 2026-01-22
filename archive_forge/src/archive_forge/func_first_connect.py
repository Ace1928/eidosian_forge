from __future__ import annotations
import typing
from typing import Any
from typing import Optional
from typing import Type
from typing import Union
from .base import ConnectionPoolEntry
from .base import Pool
from .base import PoolProxiedConnection
from .base import PoolResetState
from .. import event
from .. import util
def first_connect(self, dbapi_connection: DBAPIConnection, connection_record: ConnectionPoolEntry) -> None:
    """Called exactly once for the first time a DBAPI connection is
        checked out from a particular :class:`_pool.Pool`.

        The rationale for :meth:`_events.PoolEvents.first_connect`
        is to determine
        information about a particular series of database connections based
        on the settings used for all connections.  Since a particular
        :class:`_pool.Pool`
        refers to a single "creator" function (which in terms
        of a :class:`_engine.Engine`
        refers to the URL and connection options used),
        it is typically valid to make observations about a single connection
        that can be safely assumed to be valid about all subsequent
        connections, such as the database version, the server and client
        encoding settings, collation settings, and many others.

        :param dbapi_connection: a DBAPI connection.
         The :attr:`.ConnectionPoolEntry.dbapi_connection` attribute.

        :param connection_record: the :class:`.ConnectionPoolEntry` managing
         the DBAPI connection.

        """