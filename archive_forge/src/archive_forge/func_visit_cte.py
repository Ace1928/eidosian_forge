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
def visit_cte(self, cte: CTE, asfrom: bool=False, ashint: bool=False, fromhints: Optional[_FromHintsType]=None, visiting_cte: Optional[CTE]=None, from_linter: Optional[FromLinter]=None, cte_opts: selectable._CTEOpts=selectable._CTEOpts(False), **kwargs: Any) -> Optional[str]:
    self_ctes = self._init_cte_state()
    assert self_ctes is self.ctes
    kwargs['visiting_cte'] = cte
    cte_name = cte.name
    if isinstance(cte_name, elements._truncated_label):
        cte_name = self._truncated_identifier('alias', cte_name)
    is_new_cte = True
    embedded_in_current_named_cte = False
    _reference_cte = cte._get_reference_cte()
    nesting = cte.nesting or cte_opts.nesting
    if _reference_cte in self.level_name_by_cte:
        cte_level, _, existing_cte_opts = self.level_name_by_cte[_reference_cte]
        assert _ == cte_name
        cte_level_name = (cte_level, cte_name)
        existing_cte = self.ctes_by_level_name[cte_level_name]
        if cte_opts.nesting:
            if existing_cte_opts.nesting:
                raise exc.CompileError("CTE is stated as 'nest_here' in more than one location")
            old_level_name = (cte_level, cte_name)
            cte_level = len(self.stack) if nesting else 1
            cte_level_name = new_level_name = (cte_level, cte_name)
            del self.ctes_by_level_name[old_level_name]
            self.ctes_by_level_name[new_level_name] = existing_cte
            self.level_name_by_cte[_reference_cte] = new_level_name + (cte_opts,)
    else:
        cte_level = len(self.stack) if nesting else 1
        cte_level_name = (cte_level, cte_name)
        if cte_level_name in self.ctes_by_level_name:
            existing_cte = self.ctes_by_level_name[cte_level_name]
        else:
            existing_cte = None
    if existing_cte is not None:
        embedded_in_current_named_cte = visiting_cte is existing_cte
        if cte is existing_cte._restates or cte is existing_cte:
            is_new_cte = False
        elif existing_cte is cte._restates:
            del self_ctes[existing_cte]
            existing_cte_reference_cte = existing_cte._get_reference_cte()
            assert existing_cte_reference_cte is _reference_cte
            assert existing_cte_reference_cte is existing_cte
            del self.level_name_by_cte[existing_cte_reference_cte]
        elif (cte._is_clone_of is not None or existing_cte._is_clone_of is not None) and cte.compare(existing_cte):
            is_new_cte = False
        else:
            raise exc.CompileError('Multiple, unrelated CTEs found with the same name: %r' % cte_name)
    if not asfrom and (not is_new_cte):
        return None
    if cte._cte_alias is not None:
        pre_alias_cte = cte._cte_alias
        cte_pre_alias_name = cte._cte_alias.name
        if isinstance(cte_pre_alias_name, elements._truncated_label):
            cte_pre_alias_name = self._truncated_identifier('alias', cte_pre_alias_name)
    else:
        pre_alias_cte = cte
        cte_pre_alias_name = None
    if is_new_cte:
        self.ctes_by_level_name[cte_level_name] = cte
        self.level_name_by_cte[_reference_cte] = cte_level_name + (cte_opts,)
        if pre_alias_cte not in self.ctes:
            self.visit_cte(pre_alias_cte, **kwargs)
        if not cte_pre_alias_name and cte not in self_ctes:
            if cte.recursive:
                self.ctes_recursive = True
            text = self.preparer.format_alias(cte, cte_name)
            if cte.recursive:
                col_source = cte.element
                recur_cols = [fallback_label_name or proxy_name for _, proxy_name, fallback_label_name, c, repeated in col_source._generate_columns_plus_names(True) if not repeated]
                text += '(%s)' % ', '.join((self.preparer.format_label_name(ident, anon_map=self.anon_map) for ident in recur_cols))
            assert kwargs.get('subquery', False) is False
            if not self.stack:
                return cte.element._compiler_dispatch(self, asfrom=asfrom, **kwargs)
            else:
                prefixes = self._generate_prefixes(cte, cte._prefixes, **kwargs)
                inner = cte.element._compiler_dispatch(self, asfrom=True, **kwargs)
                text += ' AS %s\n(%s)' % (prefixes, inner)
            if cte._suffixes:
                text += ' ' + self._generate_prefixes(cte, cte._suffixes, **kwargs)
            self_ctes[cte] = text
    if asfrom:
        if from_linter:
            from_linter.froms[cte._de_clone()] = cte_name
        if not is_new_cte and embedded_in_current_named_cte:
            return self.preparer.format_alias(cte, cte_name)
        if cte_pre_alias_name:
            text = self.preparer.format_alias(cte, cte_pre_alias_name)
            if self.preparer._requires_quotes(cte_name):
                cte_name = self.preparer.quote(cte_name)
            text += self.get_render_as_alias_suffix(cte_name)
            return text
        else:
            return self.preparer.format_alias(cte, cte_name)
    return None