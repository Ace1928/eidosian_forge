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
def visit_insert(self, insert_stmt, visited_bindparam=None, visiting_cte=None, **kw):
    compile_state = insert_stmt._compile_state_factory(insert_stmt, self, **kw)
    insert_stmt = compile_state.statement
    if visiting_cte is not None:
        kw['visiting_cte'] = visiting_cte
        toplevel = False
    else:
        toplevel = not self.stack
    if toplevel:
        self.isinsert = True
        if not self.dml_compile_state:
            self.dml_compile_state = compile_state
        if not self.compile_state:
            self.compile_state = compile_state
    self.stack.append({'correlate_froms': set(), 'asfrom_froms': set(), 'selectable': insert_stmt})
    counted_bindparam = 0
    visited_bindparam = None
    if self.positional and visiting_cte is None:
        visited_bindparam = []
    crud_params_struct = crud._get_crud_params(self, insert_stmt, compile_state, toplevel, visited_bindparam=visited_bindparam, **kw)
    if self.positional and visited_bindparam is not None:
        counted_bindparam = len(visited_bindparam)
        if self._numeric_binds:
            if self._values_bindparam is not None:
                self._values_bindparam += visited_bindparam
            else:
                self._values_bindparam = visited_bindparam
    crud_params_single = crud_params_struct.single_params
    if not crud_params_single and (not self.dialect.supports_default_values) and (not self.dialect.supports_default_metavalue) and (not self.dialect.supports_empty_insert):
        raise exc.CompileError("The '%s' dialect with current database version settings does not support empty inserts." % self.dialect.name)
    if compile_state._has_multi_parameters:
        if not self.dialect.supports_multivalues_insert:
            raise exc.CompileError("The '%s' dialect with current database version settings does not support in-place multirow inserts." % self.dialect.name)
        elif (self.implicit_returning or insert_stmt._returning) and insert_stmt._sort_by_parameter_order:
            raise exc.CompileError('RETURNING cannot be determinstically sorted when using an INSERT which includes multi-row values().')
        crud_params_single = crud_params_struct.single_params
    else:
        crud_params_single = crud_params_struct.single_params
    preparer = self.preparer
    supports_default_values = self.dialect.supports_default_values
    text = 'INSERT '
    if insert_stmt._prefixes:
        text += self._generate_prefixes(insert_stmt, insert_stmt._prefixes, **kw)
    text += 'INTO '
    table_text = preparer.format_table(insert_stmt.table)
    if insert_stmt._hints:
        _, table_text = self._setup_crud_hints(insert_stmt, table_text)
    if insert_stmt._independent_ctes:
        self._dispatch_independent_ctes(insert_stmt, kw)
    text += table_text
    if crud_params_single or not supports_default_values:
        text += ' (%s)' % ', '.join([expr for _, expr, _, _ in crud_params_single])
    use_insertmanyvalues = crud_params_struct.use_insertmanyvalues
    named_sentinel_params: Optional[Sequence[str]] = None
    add_sentinel_cols = None
    implicit_sentinel = False
    returning_cols = self.implicit_returning or insert_stmt._returning
    if returning_cols:
        add_sentinel_cols = crud_params_struct.use_sentinel_columns
        if add_sentinel_cols is not None:
            assert use_insertmanyvalues
            _params_by_col = {col: param_names for col, _, _, param_names in crud_params_single}
            named_sentinel_params = []
            for _add_sentinel_col in add_sentinel_cols:
                if _add_sentinel_col not in _params_by_col:
                    named_sentinel_params = None
                    break
                param_name = self._within_exec_param_key_getter(_add_sentinel_col)
                if param_name not in _params_by_col[_add_sentinel_col]:
                    named_sentinel_params = None
                    break
                named_sentinel_params.append(param_name)
            if named_sentinel_params is None:
                if self.dialect.insertmanyvalues_implicit_sentinel & InsertmanyvaluesSentinelOpts.ANY_AUTOINCREMENT:
                    implicit_sentinel = True
                else:
                    assert not add_sentinel_cols[0]._insert_sentinel, 'sentinel selection rules should have prevented us from getting here for this dialect'
            returning_cols = list(returning_cols) + list(add_sentinel_cols)
        returning_clause = self.returning_clause(insert_stmt, returning_cols, populate_result_map=toplevel)
        if self.returning_precedes_values:
            text += ' ' + returning_clause
    else:
        returning_clause = None
    if insert_stmt.select is not None:
        select_text = self.process(self.stack[-1]['insert_from_select'], insert_into=True, **kw)
        if self.ctes and self.dialect.cte_follows_insert:
            nesting_level = len(self.stack) if not toplevel else None
            text += ' %s%s' % (self._render_cte_clause(nesting_level=nesting_level, include_following_stack=True), select_text)
        else:
            text += ' %s' % select_text
    elif not crud_params_single and supports_default_values:
        text += ' DEFAULT VALUES'
        if use_insertmanyvalues:
            self._insertmanyvalues = _InsertManyValues(True, self.dialect.default_metavalue_token, cast('List[crud._CrudParamElementStr]', crud_params_single), counted_bindparam, sort_by_parameter_order=insert_stmt._sort_by_parameter_order, includes_upsert_behaviors=insert_stmt._post_values_clause is not None, sentinel_columns=add_sentinel_cols, num_sentinel_columns=len(add_sentinel_cols) if add_sentinel_cols else 0, implicit_sentinel=implicit_sentinel)
    elif compile_state._has_multi_parameters:
        text += ' VALUES %s' % (', '.join(('(%s)' % ', '.join((value for _, _, value, _ in crud_param_set)) for crud_param_set in crud_params_struct.all_multi_params)),)
    else:
        insert_single_values_expr = ', '.join([value for _, _, value, _ in cast('List[crud._CrudParamElementStr]', crud_params_single)])
        if use_insertmanyvalues:
            if implicit_sentinel and self.dialect.insertmanyvalues_implicit_sentinel & InsertmanyvaluesSentinelOpts.USE_INSERT_FROM_SELECT and (not crud_params_struct.is_default_metavalue_only):
                embed_sentinel_value = True
                render_bind_casts = self.dialect.insertmanyvalues_implicit_sentinel & InsertmanyvaluesSentinelOpts.RENDER_SELECT_COL_CASTS
                colnames = ', '.join((f'p{i}' for i, _ in enumerate(crud_params_single)))
                if render_bind_casts:
                    colnames_w_cast = ', '.join((self.render_bind_cast(col.type, col.type._unwrapped_dialect_impl(self.dialect), f'p{i}') for i, (col, *_) in enumerate(crud_params_single)))
                else:
                    colnames_w_cast = colnames
                text += f' SELECT {colnames_w_cast} FROM (VALUES ({insert_single_values_expr})) AS imp_sen({colnames}, sen_counter) ORDER BY sen_counter'
            else:
                embed_sentinel_value = False
                text += f' VALUES ({insert_single_values_expr})'
            self._insertmanyvalues = _InsertManyValues(is_default_expr=False, single_values_expr=insert_single_values_expr, insert_crud_params=cast('List[crud._CrudParamElementStr]', crud_params_single), num_positional_params_counted=counted_bindparam, sort_by_parameter_order=insert_stmt._sort_by_parameter_order, includes_upsert_behaviors=insert_stmt._post_values_clause is not None, sentinel_columns=add_sentinel_cols, num_sentinel_columns=len(add_sentinel_cols) if add_sentinel_cols else 0, sentinel_param_keys=named_sentinel_params, implicit_sentinel=implicit_sentinel, embed_values_counter=embed_sentinel_value)
        else:
            text += f' VALUES ({insert_single_values_expr})'
    if insert_stmt._post_values_clause is not None:
        post_values_clause = self.process(insert_stmt._post_values_clause, **kw)
        if post_values_clause:
            text += ' ' + post_values_clause
    if returning_clause and (not self.returning_precedes_values):
        text += ' ' + returning_clause
    if self.ctes and (not self.dialect.cte_follows_insert):
        nesting_level = len(self.stack) if not toplevel else None
        text = self._render_cte_clause(nesting_level=nesting_level, include_following_stack=True) + text
    self.stack.pop(-1)
    return text