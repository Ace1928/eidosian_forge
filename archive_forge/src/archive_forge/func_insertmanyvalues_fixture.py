from __future__ import annotations
import itertools
import random
import re
import sys
import sqlalchemy as sa
from .base import TestBase
from .. import config
from .. import mock
from ..assertions import eq_
from ..assertions import ne_
from ..util import adict
from ..util import drop_all_tables_from_metadata
from ... import event
from ... import util
from ...schema import sort_tables_and_constraints
from ...sql import visitors
from ...sql.elements import ClauseElement
def insertmanyvalues_fixture(connection, randomize_rows=False, warn_on_downgraded=False):
    dialect = connection.dialect
    orig_dialect = dialect._deliver_insertmanyvalues_batches
    orig_conn = connection._exec_insertmany_context

    class RandomCursor:
        __slots__ = ('cursor',)

        def __init__(self, cursor):
            self.cursor = cursor

        @property
        def description(self):
            return self.cursor.description

        def fetchall(self):
            rows = self.cursor.fetchall()
            rows = list(rows)
            random.shuffle(rows)
            return rows

    def _deliver_insertmanyvalues_batches(cursor, statement, parameters, generic_setinputsizes, context):
        if randomize_rows:
            cursor = RandomCursor(cursor)
        for batch in orig_dialect(cursor, statement, parameters, generic_setinputsizes, context):
            if warn_on_downgraded and batch.is_downgraded:
                util.warn('Batches were downgraded for sorted INSERT')
            yield batch

    def _exec_insertmany_context(dialect, context):
        with mock.patch.object(dialect, '_deliver_insertmanyvalues_batches', new=_deliver_insertmanyvalues_batches):
            return orig_conn(dialect, context)
    connection._exec_insertmany_context = _exec_insertmany_context