from the proposed insertion. These values are specified using the
from __future__ import annotations
import datetime
import numbers
import re
from typing import Optional
from .json import JSON
from .json import JSONIndexType
from .json import JSONPathType
from ... import exc
from ... import schema as sa_schema
from ... import sql
from ... import text
from ... import types as sqltypes
from ... import util
from ...engine import default
from ...engine import processors
from ...engine import reflection
from ...engine.reflection import ReflectionDefaults
from ...sql import coercions
from ...sql import ColumnElement
from ...sql import compiler
from ...sql import elements
from ...sql import roles
from ...sql import schema
from ...types import BLOB  # noqa
from ...types import BOOLEAN  # noqa
from ...types import CHAR  # noqa
from ...types import DECIMAL  # noqa
from ...types import FLOAT  # noqa
from ...types import INTEGER  # noqa
from ...types import NUMERIC  # noqa
from ...types import REAL  # noqa
from ...types import SMALLINT  # noqa
from ...types import TEXT  # noqa
from ...types import TIMESTAMP  # noqa
from ...types import VARCHAR  # noqa
def get_column_specification(self, column, **kwargs):
    coltype = self.dialect.type_compiler_instance.process(column.type, type_expression=column)
    colspec = self.preparer.format_column(column) + ' ' + coltype
    default = self.get_column_default_string(column)
    if default is not None:
        if isinstance(column.server_default.arg, ColumnElement):
            default = '(' + default + ')'
        colspec += ' DEFAULT ' + default
    if not column.nullable:
        colspec += ' NOT NULL'
        on_conflict_clause = column.dialect_options['sqlite']['on_conflict_not_null']
        if on_conflict_clause is not None:
            colspec += ' ON CONFLICT ' + on_conflict_clause
    if column.primary_key:
        if column.autoincrement is True and len(column.table.primary_key.columns) != 1:
            raise exc.CompileError('SQLite does not support autoincrement for composite primary keys')
        if column.table.dialect_options['sqlite']['autoincrement'] and len(column.table.primary_key.columns) == 1 and issubclass(column.type._type_affinity, sqltypes.Integer) and (not column.foreign_keys):
            colspec += ' PRIMARY KEY'
            on_conflict_clause = column.dialect_options['sqlite']['on_conflict_primary_key']
            if on_conflict_clause is not None:
                colspec += ' ON CONFLICT ' + on_conflict_clause
            colspec += ' AUTOINCREMENT'
    if column.computed is not None:
        colspec += ' ' + self.process(column.computed)
    return colspec