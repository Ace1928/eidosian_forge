from __future__ import annotations
import contextlib
import re
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import Mapping
from typing import Optional
from typing import Protocol
from typing import Set
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from sqlalchemy import __version__
from sqlalchemy import inspect
from sqlalchemy import schema
from sqlalchemy import sql
from sqlalchemy import types as sqltypes
from sqlalchemy.engine import url
from sqlalchemy.schema import CheckConstraint
from sqlalchemy.schema import Column
from sqlalchemy.schema import ForeignKeyConstraint
from sqlalchemy.sql import visitors
from sqlalchemy.sql.base import DialectKWArgs
from sqlalchemy.sql.elements import BindParameter
from sqlalchemy.sql.elements import ColumnClause
from sqlalchemy.sql.elements import quoted_name
from sqlalchemy.sql.elements import TextClause
from sqlalchemy.sql.elements import UnaryExpression
from sqlalchemy.sql.visitors import traverse
from typing_extensions import TypeGuard
Wrap around a sqlalchemy text() construct in such a way that
    we appear like a column-oriented SQL expression to an Index
    construct.

    The issue here is that currently the Postgresql dialect, the biggest
    recipient of functional indexes, keys all the index expressions to
    the corresponding column expressions when rendering CREATE INDEX,
    so the Index we create here needs to have a .columns collection that
    is the same length as the .expressions collection.  Ultimately
    SQLAlchemy should support text() expressions in indexes.

    See SQLAlchemy issue 3174.

    