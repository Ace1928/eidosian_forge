from __future__ import annotations
import functools
import operator
import random
import re
from time import perf_counter
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import MutableSequence
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
import weakref
from . import characteristics
from . import cursor as _cursor
from . import interfaces
from .base import Connection
from .interfaces import CacheStats
from .interfaces import DBAPICursor
from .interfaces import Dialect
from .interfaces import ExecuteStyle
from .interfaces import ExecutionContext
from .reflection import ObjectKind
from .reflection import ObjectScope
from .. import event
from .. import exc
from .. import pool
from .. import util
from ..sql import compiler
from ..sql import dml
from ..sql import expression
from ..sql import type_api
from ..sql._typing import is_tuple_type
from ..sql.base import _NoArg
from ..sql.compiler import DDLCompiler
from ..sql.compiler import InsertmanyvaluesSentinelOpts
from ..sql.compiler import SQLCompiler
from ..sql.elements import quoted_name
from ..util.typing import Final
from ..util.typing import Literal
@util.memoized_property
def insert_executemany_returning_sort_by_parameter_order(self):
    """Default implementation for
        insert_executemany_returning_deterministic_order, if not otherwise
        overridden by the specific dialect.

        The default dialect determines "insert_executemany_returning" can have
        deterministic order only if the dialect in use has opted into using the
        "use_insertmanyvalues" feature, which implements deterministic ordering
        using client side sentinel columns only by default.  The
        "insertmanyvalues" feature also features alternate forms that can
        use server-generated PK values as "sentinels", but those are only
        used if the :attr:`.Dialect.insertmanyvalues_implicit_sentinel`
        bitflag enables those alternate SQL forms, which are disabled
        by default.

        If the dialect in use hasn't opted into that, then this attribute is
        False, unless the dialect in question overrides this and provides some
        other implementation (such as the Oracle dialect).

        """
    return self.insert_returning and self.use_insertmanyvalues