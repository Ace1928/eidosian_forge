import collections.abc
import dataclasses
import inspect
from typing import Any
from typing import Callable
from typing import Collection
from typing import final
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import NamedTuple
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import warnings
from .._code import getfslineno
from ..compat import ascii_escaped
from ..compat import NOTSET
from ..compat import NotSetType
from _pytest.config import Config
from _pytest.deprecated import check_ispytest
from _pytest.deprecated import MARKED_FIXTURE
from _pytest.outcomes import fail
from _pytest.warning_types import PytestUnknownMarkWarning
def get_empty_parameterset_mark(config: Config, argnames: Sequence[str], func) -> 'MarkDecorator':
    from ..nodes import Collector
    fs, lineno = getfslineno(func)
    reason = 'got empty parameter set %r, function %s at %s:%d' % (argnames, func.__name__, fs, lineno)
    requested_mark = config.getini(EMPTY_PARAMETERSET_OPTION)
    if requested_mark in ('', None, 'skip'):
        mark = MARK_GEN.skip(reason=reason)
    elif requested_mark == 'xfail':
        mark = MARK_GEN.xfail(reason=reason, run=False)
    elif requested_mark == 'fail_at_collect':
        f_name = func.__name__
        _, lineno = getfslineno(func)
        raise Collector.CollectError("Empty parameter set in '%s' at line %d" % (f_name, lineno + 1))
    else:
        raise LookupError(requested_mark)
    return mark