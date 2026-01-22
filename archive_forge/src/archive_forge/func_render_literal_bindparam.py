from __future__ import annotations
import collections
import collections.abc as collections_abc
import contextlib
from enum import IntEnum
import functools
import itertools
import operator
import re
from time import perf_counter
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import FrozenSet
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import Pattern
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from . import base
from . import coercions
from . import crud
from . import elements
from . import functions
from . import operators
from . import roles
from . import schema
from . import selectable
from . import sqltypes
from . import util as sql_util
from ._typing import is_column_element
from ._typing import is_dml
from .base import _de_clone
from .base import _from_objects
from .base import _NONE_NAME
from .base import _SentinelDefaultCharacterization
from .base import Executable
from .base import NO_ARG
from .elements import ClauseElement
from .elements import quoted_name
from .schema import Column
from .sqltypes import TupleType
from .type_api import TypeEngine
from .visitors import prefix_anon_map
from .visitors import Visitable
from .. import exc
from .. import util
from ..util import FastIntFlag
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import TypedDict
def render_literal_bindparam(self, bindparam, render_literal_value=NO_ARG, bind_expression_template=None, **kw):
    if render_literal_value is not NO_ARG:
        value = render_literal_value
    else:
        if bindparam.value is None and bindparam.callable is None:
            op = kw.get('_binary_op', None)
            if op and op not in (operators.is_, operators.is_not):
                util.warn_limited("Bound parameter '%s' rendering literal NULL in a SQL expression; comparisons to NULL should not use operators outside of 'is' or 'is not'", (bindparam.key,))
            return self.process(sqltypes.NULLTYPE, **kw)
        value = bindparam.effective_value
    if bindparam.expanding:
        leep = self._literal_execute_expanding_parameter_literal_binds
        to_update, replacement_expr = leep(bindparam, value, bind_expression_template=bind_expression_template)
        return replacement_expr
    else:
        return self.render_literal_value(value, bindparam.type)