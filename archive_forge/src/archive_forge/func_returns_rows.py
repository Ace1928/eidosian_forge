from __future__ import annotations
import collections
import functools
import operator
import typing
from typing import Any
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .result import IteratorResult
from .result import MergedResult
from .result import Result
from .result import ResultMetaData
from .result import SimpleResultMetaData
from .result import tuplegetter
from .row import Row
from .. import exc
from .. import util
from ..sql import elements
from ..sql import sqltypes
from ..sql import util as sql_util
from ..sql.base import _generative
from ..sql.compiler import ResultColumnsEntry
from ..sql.compiler import RM_NAME
from ..sql.compiler import RM_OBJECTS
from ..sql.compiler import RM_RENDERED_NAME
from ..sql.compiler import RM_TYPE
from ..sql.type_api import TypeEngine
from ..util import compat
from ..util.typing import Literal
from ..util.typing import Self
@property
def returns_rows(self):
    """True if this :class:`_engine.CursorResult` returns zero or more
        rows.

        I.e. if it is legal to call the methods
        :meth:`_engine.CursorResult.fetchone`,
        :meth:`_engine.CursorResult.fetchmany`
        :meth:`_engine.CursorResult.fetchall`.

        Overall, the value of :attr:`_engine.CursorResult.returns_rows` should
        always be synonymous with whether or not the DBAPI cursor had a
        ``.description`` attribute, indicating the presence of result columns,
        noting that a cursor that returns zero rows still has a
        ``.description`` if a row-returning statement was emitted.

        This attribute should be True for all results that are against
        SELECT statements, as well as for DML statements INSERT/UPDATE/DELETE
        that use RETURNING.   For INSERT/UPDATE/DELETE statements that were
        not using RETURNING, the value will usually be False, however
        there are some dialect-specific exceptions to this, such as when
        using the MSSQL / pyodbc dialect a SELECT is emitted inline in
        order to retrieve an inserted primary key value.


        """
    return self._metadata.returns_rows