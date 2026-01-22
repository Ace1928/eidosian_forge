from __future__ import annotations
import contextlib
from enum import Enum
import itertools
import sys
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
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
import weakref
from . import attributes
from . import bulk_persistence
from . import context
from . import descriptor_props
from . import exc
from . import identity
from . import loading
from . import query
from . import state as statelib
from ._typing import _O
from ._typing import insp_is_mapper
from ._typing import is_composite_class
from ._typing import is_orm_option
from ._typing import is_user_defined_option
from .base import _class_to_mapper
from .base import _none_set
from .base import _state_mapper
from .base import instance_str
from .base import LoaderCallableStatus
from .base import object_mapper
from .base import object_state
from .base import PassiveFlag
from .base import state_str
from .context import FromStatement
from .context import ORMCompileState
from .identity import IdentityMap
from .query import Query
from .state import InstanceState
from .state_changes import _StateChange
from .state_changes import _StateChangeState
from .state_changes import _StateChangeStates
from .unitofwork import UOWTransaction
from .. import engine
from .. import exc as sa_exc
from .. import sql
from .. import util
from ..engine import Connection
from ..engine import Engine
from ..engine.util import TransactionalContext
from ..event import dispatcher
from ..event import EventTarget
from ..inspection import inspect
from ..inspection import Inspectable
from ..sql import coercions
from ..sql import dml
from ..sql import roles
from ..sql import Select
from ..sql import TableClause
from ..sql import visitors
from ..sql.base import _NoArg
from ..sql.base import CompileState
from ..sql.schema import Table
from ..sql.selectable import ForUpdateArg
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..util import IdentitySet
from ..util.typing import Literal
from ..util.typing import Protocol
def make_transient_to_detached(instance: object) -> None:
    """Make the given transient instance :term:`detached`.

    .. note::

        :func:`.make_transient_to_detached` is a special-case function for
        advanced use cases only.

    All attribute history on the given instance
    will be reset as though the instance were freshly loaded
    from a query.  Missing attributes will be marked as expired.
    The primary key attributes of the object, which are required, will be made
    into the "key" of the instance.

    The object can then be added to a session, or merged
    possibly with the load=False flag, at which point it will look
    as if it were loaded that way, without emitting SQL.

    This is a special use case function that differs from a normal
    call to :meth:`.Session.merge` in that a given persistent state
    can be manufactured without any SQL calls.

    .. seealso::

        :func:`.make_transient`

        :meth:`.Session.enable_relationship_loading`

    """
    state = attributes.instance_state(instance)
    if state.session_id or state.key:
        raise sa_exc.InvalidRequestError('Given object must be transient')
    state.key = state.mapper._identity_key_from_state(state)
    if state._deleted:
        del state._deleted
    state._commit_all(state.dict)
    state._expire_attributes(state.dict, state.unloaded)