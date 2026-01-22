from __future__ import annotations
import collections
from enum import Enum
import itertools
from itertools import zip_longest
import operator
import re
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import roles
from . import visitors
from .cache_key import HasCacheKey  # noqa
from .cache_key import MemoizedHasCacheKey  # noqa
from .traversals import HasCopyInternals  # noqa
from .visitors import ClauseVisitor
from .visitors import ExtendedInternalTraversal
from .visitors import ExternallyTraversible
from .visitors import InternalTraversal
from .. import event
from .. import exc
from .. import util
from ..util import HasMemoized as HasMemoized
from ..util import hybridmethod
from ..util import typing as compat_typing
from ..util.typing import Protocol
from ..util.typing import Self
from ..util.typing import TypeGuard
@classmethod
def from_execution_options(cls, key, attrs, exec_options, statement_exec_options):
    """process Options argument in terms of execution options.


        e.g.::

            (
                load_options,
                execution_options,
            ) = QueryContext.default_load_options.from_execution_options(
                "_sa_orm_load_options",
                {
                    "populate_existing",
                    "autoflush",
                    "yield_per"
                },
                execution_options,
                statement._execution_options,
            )

        get back the Options and refresh "_sa_orm_load_options" in the
        exec options dict w/ the Options as well

        """
    check_argnames = attrs.intersection(set(exec_options).union(statement_exec_options))
    existing_options = exec_options.get(key, cls)
    if check_argnames:
        result = {}
        for argname in check_argnames:
            local = '_' + argname
            if argname in exec_options:
                result[local] = exec_options[argname]
            elif argname in statement_exec_options:
                result[local] = statement_exec_options[argname]
        new_options = existing_options + result
        exec_options = util.immutabledict().merge_with(exec_options, {key: new_options})
        return (new_options, exec_options)
    else:
        return (existing_options, exec_options)