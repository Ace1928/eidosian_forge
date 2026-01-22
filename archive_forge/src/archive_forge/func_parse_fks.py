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
def parse_fks():
    if table_data is None:
        return
    FK_PATTERN = '(?:CONSTRAINT (\\w+) +)?FOREIGN KEY *\\( *(.+?) *\\) +REFERENCES +(?:(?:"(.+?)")|([a-z0-9_]+)) *\\( *((?:(?:"[^"]+"|[a-z0-9_]+) *(?:, *)?)+)\\) *((?:ON (?:DELETE|UPDATE) (?:SET NULL|SET DEFAULT|CASCADE|RESTRICT|NO ACTION) *)*)((?:NOT +)?DEFERRABLE)?(?: +INITIALLY +(DEFERRED|IMMEDIATE))?'
    for match in re.finditer(FK_PATTERN, table_data, re.I):
        constraint_name, constrained_columns, referred_quoted_name, referred_name, referred_columns, onupdatedelete, deferrable, initially = match.group(1, 2, 3, 4, 5, 6, 7, 8)
        constrained_columns = list(self._find_cols_in_sig(constrained_columns))
        if not referred_columns:
            referred_columns = constrained_columns
        else:
            referred_columns = list(self._find_cols_in_sig(referred_columns))
        referred_name = referred_quoted_name or referred_name
        options = {}
        for token in re.split(' *\\bON\\b *', onupdatedelete.upper()):
            if token.startswith('DELETE'):
                ondelete = token[6:].strip()
                if ondelete and ondelete != 'NO ACTION':
                    options['ondelete'] = ondelete
            elif token.startswith('UPDATE'):
                onupdate = token[6:].strip()
                if onupdate and onupdate != 'NO ACTION':
                    options['onupdate'] = onupdate
        if deferrable:
            options['deferrable'] = 'NOT' not in deferrable.upper()
        if initially:
            options['initially'] = initially.upper()
        yield (constraint_name, constrained_columns, referred_name, referred_columns, options)