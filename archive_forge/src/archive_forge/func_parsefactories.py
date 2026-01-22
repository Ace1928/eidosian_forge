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
def parsefactories(self, node_or_obj: Union[nodes.Node, object], nodeid: Union[str, NotSetType, None]=NOTSET, *, unittest: bool=False) -> None:
    """Collect fixtures from a collection node or object.

        Found fixtures are parsed into `FixtureDef`s and saved.

        If `node_or_object` is a collection node (with an underlying Python
        object), the node's object is traversed and the node's nodeid is used to
        determine the fixtures' visibilty. `nodeid` must not be specified in
        this case.

        If `node_or_object` is an object (e.g. a plugin), the object is
        traversed and the given `nodeid` is used to determine the fixtures'
        visibility. `nodeid` must be specified in this case; None and "" mean
        total visibility.
        """
    if nodeid is not NOTSET:
        holderobj = node_or_obj
    else:
        assert isinstance(node_or_obj, nodes.Node)
        holderobj = cast(object, node_or_obj.obj)
        assert isinstance(node_or_obj.nodeid, str)
        nodeid = node_or_obj.nodeid
    if holderobj in self._holderobjseen:
        return
    self._holderobjseen.add(holderobj)
    for name in dir(holderobj):
        obj = safe_getattr(holderobj, name, None)
        marker = getfixturemarker(obj)
        if not isinstance(marker, FixtureFunctionMarker):
            continue
        if marker.name:
            name = marker.name
        func = get_real_method(obj, holderobj)
        self._register_fixture(name=name, nodeid=nodeid, func=func, scope=marker.scope, params=marker.params, unittest=unittest, ids=marker.ids, autouse=marker.autouse)