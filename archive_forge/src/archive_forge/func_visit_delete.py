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
def visit_delete(self, delete_stmt, visiting_cte=None, **kw):
    compile_state = delete_stmt._compile_state_factory(delete_stmt, self, **kw)
    delete_stmt = compile_state.statement
    if visiting_cte is not None:
        kw['visiting_cte'] = visiting_cte
        toplevel = False
    else:
        toplevel = not self.stack
    if toplevel:
        self.isdelete = True
        if not self.dml_compile_state:
            self.dml_compile_state = compile_state
        if not self.compile_state:
            self.compile_state = compile_state
    if self.linting & COLLECT_CARTESIAN_PRODUCTS:
        from_linter = FromLinter({}, set())
        warn_linting = self.linting & WARN_LINTING
        if toplevel:
            self.from_linter = from_linter
    else:
        from_linter = None
        warn_linting = False
    extra_froms = compile_state._extra_froms
    correlate_froms = {delete_stmt.table}.union(extra_froms)
    self.stack.append({'correlate_froms': correlate_froms, 'asfrom_froms': correlate_froms, 'selectable': delete_stmt})
    text = 'DELETE '
    if delete_stmt._prefixes:
        text += self._generate_prefixes(delete_stmt, delete_stmt._prefixes, **kw)
    text += 'FROM '
    try:
        table_text = self.delete_table_clause(delete_stmt, delete_stmt.table, extra_froms, from_linter=from_linter)
    except TypeError:
        table_text = self.delete_table_clause(delete_stmt, delete_stmt.table, extra_froms)
        if from_linter:
            _ = self.process(delete_stmt.table, from_linter=from_linter)
    crud._get_crud_params(self, delete_stmt, compile_state, toplevel, **kw)
    if delete_stmt._hints:
        dialect_hints, table_text = self._setup_crud_hints(delete_stmt, table_text)
    else:
        dialect_hints = None
    if delete_stmt._independent_ctes:
        self._dispatch_independent_ctes(delete_stmt, kw)
    text += table_text
    if (self.implicit_returning or delete_stmt._returning) and self.returning_precedes_values:
        text += ' ' + self.returning_clause(delete_stmt, self.implicit_returning or delete_stmt._returning, populate_result_map=toplevel)
    if extra_froms:
        extra_from_text = self.delete_extra_from_clause(delete_stmt, delete_stmt.table, extra_froms, dialect_hints, from_linter=from_linter, **kw)
        if extra_from_text:
            text += ' ' + extra_from_text
    if delete_stmt._where_criteria:
        t = self._generate_delimited_and_list(delete_stmt._where_criteria, from_linter=from_linter, **kw)
        if t:
            text += ' WHERE ' + t
    if (self.implicit_returning or delete_stmt._returning) and (not self.returning_precedes_values):
        text += ' ' + self.returning_clause(delete_stmt, self.implicit_returning or delete_stmt._returning, populate_result_map=toplevel)
    if self.ctes:
        nesting_level = len(self.stack) if not toplevel else None
        text = self._render_cte_clause(nesting_level=nesting_level) + text
    if warn_linting:
        assert from_linter is not None
        from_linter.warn(stmt_type='DELETE')
    self.stack.pop(-1)
    return text