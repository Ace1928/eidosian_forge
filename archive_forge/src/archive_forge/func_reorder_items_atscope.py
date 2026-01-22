import abc
from collections import defaultdict
from collections import deque
import dataclasses
import functools
import inspect
import os
from pathlib import Path
import sys
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Final
from typing import final
from typing import Generator
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import MutableMapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import warnings
import _pytest
from _pytest import nodes
from _pytest._code import getfslineno
from _pytest._code.code import FormattedExcinfo
from _pytest._code.code import TerminalRepr
from _pytest._io import TerminalWriter
from _pytest.compat import _PytestWrapper
from _pytest.compat import assert_never
from _pytest.compat import get_real_func
from _pytest.compat import get_real_method
from _pytest.compat import getfuncargnames
from _pytest.compat import getimfunc
from _pytest.compat import getlocation
from _pytest.compat import is_generator
from _pytest.compat import NOTSET
from _pytest.compat import NotSetType
from _pytest.compat import safe_getattr
from _pytest.config import _PluggyPlugin
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.deprecated import MARKED_FIXTURE
from _pytest.deprecated import YIELD_FIXTURE
from _pytest.mark import Mark
from _pytest.mark import ParameterSet
from _pytest.mark.structures import MarkDecorator
from _pytest.outcomes import fail
from _pytest.outcomes import skip
from _pytest.outcomes import TEST_OUTCOME
from _pytest.pathlib import absolutepath
from _pytest.pathlib import bestrelpath
from _pytest.scope import _ScopeName
from _pytest.scope import HIGH_SCOPES
from _pytest.scope import Scope
def reorder_items_atscope(items: Dict[nodes.Item, None], argkeys_cache: Dict[Scope, Dict[nodes.Item, Dict[FixtureArgKey, None]]], items_by_argkey: Dict[Scope, Dict[FixtureArgKey, 'Deque[nodes.Item]']], scope: Scope) -> Dict[nodes.Item, None]:
    if scope is Scope.Function or len(items) < 3:
        return items
    ignore: Set[Optional[FixtureArgKey]] = set()
    items_deque = deque(items)
    items_done: Dict[nodes.Item, None] = {}
    scoped_items_by_argkey = items_by_argkey[scope]
    scoped_argkeys_cache = argkeys_cache[scope]
    while items_deque:
        no_argkey_group: Dict[nodes.Item, None] = {}
        slicing_argkey = None
        while items_deque:
            item = items_deque.popleft()
            if item in items_done or item in no_argkey_group:
                continue
            argkeys = dict.fromkeys((k for k in scoped_argkeys_cache.get(item, []) if k not in ignore), None)
            if not argkeys:
                no_argkey_group[item] = None
            else:
                slicing_argkey, _ = argkeys.popitem()
                matching_items = [i for i in scoped_items_by_argkey[slicing_argkey] if i in items]
                for i in reversed(matching_items):
                    fix_cache_order(i, argkeys_cache, items_by_argkey)
                    items_deque.appendleft(i)
                break
        if no_argkey_group:
            no_argkey_group = reorder_items_atscope(no_argkey_group, argkeys_cache, items_by_argkey, scope.next_lower())
            for item in no_argkey_group:
                items_done[item] = None
        ignore.add(slicing_argkey)
    return items_done