from __future__ import annotations
from contextlib import contextmanager
from contextlib import nullcontext
import logging
import sys
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import ContextManager
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import Column
from sqlalchemy import literal_column
from sqlalchemy import MetaData
from sqlalchemy import PrimaryKeyConstraint
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy.engine import Engine
from sqlalchemy.engine import url as sqla_url
from sqlalchemy.engine.strategies import MockEngineStrategy
from .. import ddl
from .. import util
from ..util import sqla_compat
from ..util.compat import EncodedIO
@contextmanager
def autocommit_block(self) -> Iterator[None]:
    """Enter an "autocommit" block, for databases that support AUTOCOMMIT
        isolation levels.

        This special directive is intended to support the occasional database
        DDL or system operation that specifically has to be run outside of
        any kind of transaction block.   The PostgreSQL database platform
        is the most common target for this style of operation, as many
        of its DDL operations must be run outside of transaction blocks, even
        though the database overall supports transactional DDL.

        The method is used as a context manager within a migration script, by
        calling on :meth:`.Operations.get_context` to retrieve the
        :class:`.MigrationContext`, then invoking
        :meth:`.MigrationContext.autocommit_block` using the ``with:``
        statement::

            def upgrade():
                with op.get_context().autocommit_block():
                    op.execute("ALTER TYPE mood ADD VALUE 'soso'")

        Above, a PostgreSQL "ALTER TYPE..ADD VALUE" directive is emitted,
        which must be run outside of a transaction block at the database level.
        The :meth:`.MigrationContext.autocommit_block` method makes use of the
        SQLAlchemy ``AUTOCOMMIT`` isolation level setting, which against the
        psycogp2 DBAPI corresponds to the ``connection.autocommit`` setting,
        to ensure that the database driver is not inside of a DBAPI level
        transaction block.

        .. warning::

            As is necessary, **the database transaction preceding the block is
            unconditionally committed**.  This means that the run of migrations
            preceding the operation will be committed, before the overall
            migration operation is complete.

            It is recommended that when an application includes migrations with
            "autocommit" blocks, that
            :paramref:`.EnvironmentContext.transaction_per_migration` be used
            so that the calling environment is tuned to expect short per-file
            migrations whether or not one of them has an autocommit block.


        """
    _in_connection_transaction = self._in_connection_transaction()
    if self.impl.transactional_ddl and self.as_sql:
        self.impl.emit_commit()
    elif _in_connection_transaction:
        assert self._transaction is not None
        self._transaction.commit()
        self._transaction = None
    if not self.as_sql:
        assert self.connection is not None
        current_level = self.connection.get_isolation_level()
        base_connection = self.connection
        self.connection = self.impl.connection = base_connection.execution_options(isolation_level='AUTOCOMMIT')
        fake_trans: Optional[Transaction] = self.connection.begin()
    else:
        fake_trans = None
    try:
        yield
    finally:
        if not self.as_sql:
            assert self.connection is not None
            if fake_trans is not None:
                fake_trans.commit()
            self.connection.execution_options(isolation_level=current_level)
            self.connection = self.impl.connection = base_connection
        if self.impl.transactional_ddl and self.as_sql:
            self.impl.emit_begin()
        elif _in_connection_transaction:
            assert self.connection is not None
            self._transaction = self.connection.begin()