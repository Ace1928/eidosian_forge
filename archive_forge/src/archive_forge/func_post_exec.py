from __future__ import annotations
import codecs
import datetime
import operator
import re
from typing import overload
from typing import TYPE_CHECKING
from uuid import UUID as _python_UUID
from . import information_schema as ischema
from .json import JSON
from .json import JSONIndexType
from .json import JSONPathType
from ... import exc
from ... import Identity
from ... import schema as sa_schema
from ... import Sequence
from ... import sql
from ... import text
from ... import util
from ...engine import cursor as _cursor
from ...engine import default
from ...engine import reflection
from ...engine.reflection import ReflectionDefaults
from ...sql import coercions
from ...sql import compiler
from ...sql import elements
from ...sql import expression
from ...sql import func
from ...sql import quoted_name
from ...sql import roles
from ...sql import sqltypes
from ...sql import try_cast as try_cast  # noqa: F401
from ...sql import util as sql_util
from ...sql._typing import is_sql_compiler
from ...sql.compiler import InsertmanyvaluesSentinelOpts
from ...sql.elements import TryCast as TryCast  # noqa: F401
from ...types import BIGINT
from ...types import BINARY
from ...types import CHAR
from ...types import DATE
from ...types import DATETIME
from ...types import DECIMAL
from ...types import FLOAT
from ...types import INTEGER
from ...types import NCHAR
from ...types import NUMERIC
from ...types import NVARCHAR
from ...types import SMALLINT
from ...types import TEXT
from ...types import VARCHAR
from ...util import update_wrapper
from ...util.typing import Literal
from
from
def post_exec(self):
    """Disable IDENTITY_INSERT if enabled."""
    conn = self.root_connection
    if self.isinsert or self.isupdate or self.isdelete:
        self._rowcount = self.cursor.rowcount
    if self._select_lastrowid:
        if self.dialect.use_scope_identity:
            conn._cursor_execute(self.cursor, 'SELECT scope_identity() AS lastrowid', (), self)
        else:
            conn._cursor_execute(self.cursor, 'SELECT @@identity AS lastrowid', (), self)
        row = self.cursor.fetchall()[0]
        self._lastrowid = int(row[0])
        self.cursor_fetch_strategy = _cursor._NO_CURSOR_DML
    elif self.compiled is not None and is_sql_compiler(self.compiled) and self.compiled.effective_returning:
        self.cursor_fetch_strategy = _cursor.FullyBufferedCursorFetchStrategy(self.cursor, self.cursor.description, self.cursor.fetchall())
    if self._enable_identity_insert:
        if TYPE_CHECKING:
            assert is_sql_compiler(self.compiled)
            assert isinstance(self.compiled.compile_state, DMLState)
            assert isinstance(self.compiled.compile_state.dml_table, TableClause)
        conn._cursor_execute(self.cursor, self._opt_encode('SET IDENTITY_INSERT %s OFF' % self.identifier_preparer.format_table(self.compiled.compile_state.dml_table)), (), self)