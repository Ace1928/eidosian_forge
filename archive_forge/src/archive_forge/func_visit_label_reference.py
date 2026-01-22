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
def visit_label_reference(self, element, within_columns_clause=False, **kwargs):
    if self.stack and self.dialect.supports_simple_order_by_label:
        try:
            compile_state = cast('Union[SelectState, CompoundSelectState]', self.stack[-1]['compile_state'])
        except KeyError as ke:
            raise exc.CompileError("Can't resolve label reference for ORDER BY / GROUP BY / DISTINCT etc.") from ke
        with_cols, only_froms, only_cols = compile_state._label_resolve_dict
        if within_columns_clause:
            resolve_dict = only_froms
        else:
            resolve_dict = only_cols
        order_by_elem = element.element._order_by_label_element
        if order_by_elem is not None and order_by_elem.name in resolve_dict and order_by_elem.shares_lineage(resolve_dict[order_by_elem.name]):
            kwargs['render_label_as_label'] = element.element._order_by_label_element
    return self.process(element.element, within_columns_clause=within_columns_clause, **kwargs)