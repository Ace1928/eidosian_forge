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
@property
def current_executable(self):
    """Return the current 'executable' that is being compiled.

        This is currently the :class:`_sql.Select`, :class:`_sql.Insert`,
        :class:`_sql.Update`, :class:`_sql.Delete`,
        :class:`_sql.CompoundSelect` object that is being compiled.
        Specifically it's assigned to the ``self.stack`` list of elements.

        When a statement like the above is being compiled, it normally
        is also assigned to the ``.statement`` attribute of the
        :class:`_sql.Compiler` object.   However, all SQL constructs are
        ultimately nestable, and this attribute should never be consulted
        by a ``visit_`` method, as it is not guaranteed to be assigned
        nor guaranteed to correspond to the current statement being compiled.

        .. versionadded:: 1.3.21

            For compatibility with previous versions, use the following
            recipe::

                statement = getattr(self, "current_executable", False)
                if statement is False:
                    statement = self.stack[-1]["selectable"]

            For versions 1.4 and above, ensure only .current_executable
            is used; the format of "self.stack" may change.


        """
    try:
        return self.stack[-1]['selectable']
    except IndexError as ie:
        raise IndexError('Compiler does not have a stack entry') from ie