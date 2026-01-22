from __future__ import annotations
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import attributes
from . import exc as orm_exc
from . import path_registry
from .base import _DEFER_FOR_STATE
from .base import _RAISE_FOR_STATE
from .base import _SET_DEFERRED_EXPIRED
from .base import PassiveFlag
from .context import FromStatement
from .context import ORMCompileState
from .context import QueryContext
from .util import _none_set
from .util import state_str
from .. import exc as sa_exc
from .. import util
from ..engine import result_tuple
from ..engine.result import ChunkedIteratorResult
from ..engine.result import FrozenResult
from ..engine.result import SimpleResultMetaData
from ..sql import select
from ..sql import util as sql_util
from ..sql.selectable import ForUpdateArg
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..sql.selectable import SelectState
from ..util import EMPTY_DICT
def load_scalar_attributes(mapper, state, attribute_names, passive):
    """initiate a column-based attribute refresh operation."""
    session = state.session
    if not session:
        raise orm_exc.DetachedInstanceError('Instance %s is not bound to a Session; attribute refresh operation cannot proceed' % state_str(state))
    no_autoflush = bool(passive & attributes.NO_AUTOFLUSH)
    if attribute_names:
        attribute_names = attribute_names.intersection(mapper.attrs.keys())
    if mapper.inherits and (not mapper.concrete):
        statement = mapper._optimized_get_statement(state, attribute_names)
        if statement is not None:
            stmt = FromStatement(mapper, statement)
            return load_on_ident(session, stmt, None, only_load_props=attribute_names, refresh_state=state, no_autoflush=no_autoflush)
    has_key = bool(state.key)
    if has_key:
        identity_key = state.key
    else:
        pk_attrs = [mapper._columntoproperty[col].key for col in mapper.primary_key]
        if state.expired_attributes.intersection(pk_attrs):
            raise sa_exc.InvalidRequestError("Instance %s cannot be refreshed - it's not  persistent and does not contain a full primary key." % state_str(state))
        identity_key = mapper._identity_key_from_state(state)
    if _none_set.issubset(identity_key) and (not mapper.allow_partial_pks) or _none_set.issuperset(identity_key):
        util.warn_limited("Instance %s to be refreshed doesn't contain a full primary key - can't be refreshed (and shouldn't be expired, either).", state_str(state))
        return
    result = load_on_ident(session, select(mapper).set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL), identity_key, refresh_state=state, only_load_props=attribute_names, no_autoflush=no_autoflush)
    if has_key and result is None:
        raise orm_exc.ObjectDeletedError(state)