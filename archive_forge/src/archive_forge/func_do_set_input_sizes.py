from __future__ import annotations
import decimal
import random
import re
from . import base as oracle
from .base import OracleCompiler
from .base import OracleDialect
from .base import OracleExecutionContext
from .types import _OracleDateLiteralRender
from ... import exc
from ... import util
from ...engine import cursor as _cursor
from ...engine import interfaces
from ...engine import processors
from ...sql import sqltypes
from ...sql._typing import is_sql_compiler
def do_set_input_sizes(self, cursor, list_of_tuples, context):
    if self.positional:
        cursor.setinputsizes(*[dbtype for key, dbtype, sqltype in list_of_tuples])
    else:
        collection = ((key, dbtype) for key, dbtype, sqltype in list_of_tuples if dbtype)
        cursor.setinputsizes(**{key: dbtype for key, dbtype in collection})