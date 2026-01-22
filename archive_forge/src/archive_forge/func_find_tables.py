from __future__ import annotations
from collections import deque
import copy
from itertools import chain
import typing
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import operators
from . import roles
from . import visitors
from ._typing import is_text_clause
from .annotation import _deep_annotate as _deep_annotate  # noqa: F401
from .annotation import _deep_deannotate as _deep_deannotate  # noqa: F401
from .annotation import _shallow_annotate as _shallow_annotate  # noqa: F401
from .base import _expand_cloned
from .base import _from_objects
from .cache_key import HasCacheKey as HasCacheKey  # noqa: F401
from .ddl import sort_tables as sort_tables  # noqa: F401
from .elements import _find_columns as _find_columns
from .elements import _label_reference
from .elements import _textual_label_reference
from .elements import BindParameter
from .elements import ClauseElement
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import Grouping
from .elements import KeyedColumnElement
from .elements import Label
from .elements import NamedColumn
from .elements import Null
from .elements import UnaryExpression
from .schema import Column
from .selectable import Alias
from .selectable import FromClause
from .selectable import FromGrouping
from .selectable import Join
from .selectable import ScalarSelect
from .selectable import SelectBase
from .selectable import TableClause
from .visitors import _ET
from .. import exc
from .. import util
from ..util.typing import Literal
from ..util.typing import Protocol
def find_tables(clause: ClauseElement, *, check_columns: bool=False, include_aliases: bool=False, include_joins: bool=False, include_selects: bool=False, include_crud: bool=False) -> List[TableClause]:
    """locate Table objects within the given expression."""
    tables: List[TableClause] = []
    _visitors: Dict[str, _TraverseCallableType[Any]] = {}
    if include_selects:
        _visitors['select'] = _visitors['compound_select'] = tables.append
    if include_joins:
        _visitors['join'] = tables.append
    if include_aliases:
        _visitors['alias'] = _visitors['subquery'] = _visitors['tablesample'] = _visitors['lateral'] = tables.append
    if include_crud:
        _visitors['insert'] = _visitors['update'] = _visitors['delete'] = lambda ent: tables.append(ent.table)
    if check_columns:

        def visit_column(column):
            tables.append(column.table)
        _visitors['column'] = visit_column
    _visitors['table'] = tables.append
    visitors.traverse(clause, {}, _visitors)
    return tables