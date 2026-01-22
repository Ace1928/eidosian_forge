import collections.abc as collections_abc
import itertools
from .. import AssertsCompiledSQL
from .. import AssertsExecutionResults
from .. import config
from .. import fixtures
from ..assertions import assert_raises
from ..assertions import eq_
from ..assertions import in_
from ..assertsql import CursorSQL
from ..schema import Column
from ..schema import Table
from ... import bindparam
from ... import case
from ... import column
from ... import Computed
from ... import exists
from ... import false
from ... import ForeignKey
from ... import func
from ... import Identity
from ... import Integer
from ... import literal
from ... import literal_column
from ... import null
from ... import select
from ... import String
from ... import table
from ... import testing
from ... import text
from ... import true
from ... import tuple_
from ... import TupleType
from ... import union
from ... import values
from ...exc import DatabaseError
from ...exc import ProgrammingError
def test_execute_expanding_plus_literal_execute(self):
    table = self.tables.some_table
    stmt = select(table.c.id).where(table.c.x.in_(bindparam('q', expanding=True, literal_execute=True)))
    with self.sql_execution_asserter() as asserter:
        with config.db.connect() as conn:
            conn.execute(stmt, dict(q=[5, 6, 7]))
    asserter.assert_(CursorSQL('SELECT some_table.id \nFROM some_table \nWHERE some_table.x IN (5, 6, 7)', () if config.db.dialect.positional else {}))