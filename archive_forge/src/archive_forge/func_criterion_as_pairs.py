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
def criterion_as_pairs(expression, consider_as_foreign_keys=None, consider_as_referenced_keys=None, any_operator=False):
    """traverse an expression and locate binary criterion pairs."""
    if consider_as_foreign_keys and consider_as_referenced_keys:
        raise exc.ArgumentError("Can only specify one of 'consider_as_foreign_keys' or 'consider_as_referenced_keys'")

    def col_is(a, b):
        return a.compare(b)

    def visit_binary(binary):
        if not any_operator and binary.operator is not operators.eq:
            return
        if not isinstance(binary.left, ColumnElement) or not isinstance(binary.right, ColumnElement):
            return
        if consider_as_foreign_keys:
            if binary.left in consider_as_foreign_keys and (col_is(binary.right, binary.left) or binary.right not in consider_as_foreign_keys):
                pairs.append((binary.right, binary.left))
            elif binary.right in consider_as_foreign_keys and (col_is(binary.left, binary.right) or binary.left not in consider_as_foreign_keys):
                pairs.append((binary.left, binary.right))
        elif consider_as_referenced_keys:
            if binary.left in consider_as_referenced_keys and (col_is(binary.right, binary.left) or binary.right not in consider_as_referenced_keys):
                pairs.append((binary.left, binary.right))
            elif binary.right in consider_as_referenced_keys and (col_is(binary.left, binary.right) or binary.left not in consider_as_referenced_keys):
                pairs.append((binary.right, binary.left))
        elif isinstance(binary.left, Column) and isinstance(binary.right, Column):
            if binary.left.references(binary.right):
                pairs.append((binary.right, binary.left))
            elif binary.right.references(binary.left):
                pairs.append((binary.left, binary.right))
    pairs: List[Tuple[ColumnElement[Any], ColumnElement[Any]]] = []
    visitors.traverse(expression, {}, {'binary': visit_binary})
    return pairs