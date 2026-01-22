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
def enable_relationship_loading(self, obj: object) -> None:
    """Associate an object with this :class:`.Session` for related
        object loading.

        .. warning::

            :meth:`.enable_relationship_loading` exists to serve special
            use cases and is not recommended for general use.

        Accesses of attributes mapped with :func:`_orm.relationship`
        will attempt to load a value from the database using this
        :class:`.Session` as the source of connectivity.  The values
        will be loaded based on foreign key and primary key values
        present on this object - if not present, then those relationships
        will be unavailable.

        The object will be attached to this session, but will
        **not** participate in any persistence operations; its state
        for almost all purposes will remain either "transient" or
        "detached", except for the case of relationship loading.

        Also note that backrefs will often not work as expected.
        Altering a relationship-bound attribute on the target object
        may not fire off a backref event, if the effective value
        is what was already loaded from a foreign-key-holding value.

        The :meth:`.Session.enable_relationship_loading` method is
        similar to the ``load_on_pending`` flag on :func:`_orm.relationship`.
        Unlike that flag, :meth:`.Session.enable_relationship_loading` allows
        an object to remain transient while still being able to load
        related items.

        To make a transient object associated with a :class:`.Session`
        via :meth:`.Session.enable_relationship_loading` pending, add
        it to the :class:`.Session` using :meth:`.Session.add` normally.
        If the object instead represents an existing identity in the database,
        it should be merged using :meth:`.Session.merge`.

        :meth:`.Session.enable_relationship_loading` does not improve
        behavior when the ORM is used normally - object references should be
        constructed at the object level, not at the foreign key level, so
        that they are present in an ordinary way before flush()
        proceeds.  This method is not intended for general use.

        .. seealso::

            :paramref:`_orm.relationship.load_on_pending` - this flag
            allows per-relationship loading of many-to-ones on items that
            are pending.

            :func:`.make_transient_to_detached` - allows for an object to
            be added to a :class:`.Session` without SQL emitted, which then
            will unexpire attributes on access.

        """
    try:
        state = attributes.instance_state(obj)
    except exc.NO_STATE as err:
        raise exc.UnmappedInstanceError(obj) from err
    to_attach = self._before_attach(state, obj)
    state._load_pending = True
    if to_attach:
        self._after_attach(state, obj)