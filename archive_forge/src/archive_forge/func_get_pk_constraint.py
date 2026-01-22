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
@reflection.cache
def get_pk_constraint(self, connection, table_name, schema=None, **kw):
    constraint_name = None
    table_data = self._get_table_sql(connection, table_name, schema=schema)
    if table_data:
        PK_PATTERN = 'CONSTRAINT (\\w+) PRIMARY KEY'
        result = re.search(PK_PATTERN, table_data, re.I)
        constraint_name = result.group(1) if result else None
    cols = self.get_columns(connection, table_name, schema, **kw)
    cols = [col for col in cols if col.get('primary_key', 0) > 0]
    cols.sort(key=lambda col: col.get('primary_key'))
    pkeys = [col['name'] for col in cols]
    if pkeys:
        return {'constrained_columns': pkeys, 'name': constraint_name}
    else:
        return ReflectionDefaults.pk_constraint()