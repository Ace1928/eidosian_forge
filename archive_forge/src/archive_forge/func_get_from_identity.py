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
def get_from_identity(session: Session, mapper: Mapper[_O], key: _IdentityKeyType[_O], passive: PassiveFlag) -> Union[LoaderCallableStatus, Optional[_O]]:
    """Look up the given key in the given session's identity map,
    check the object for expired state if found.

    """
    instance = session.identity_map.get(key)
    if instance is not None:
        state = attributes.instance_state(instance)
        if mapper.inherits and (not state.mapper.isa(mapper)):
            return attributes.PASSIVE_CLASS_MISMATCH
        if state.expired:
            if not passive & attributes.SQL_OK:
                return attributes.PASSIVE_NO_RESULT
            elif not passive & attributes.RELATED_OBJECT_OK:
                return instance
            try:
                state._load_expired(state, passive)
            except orm_exc.ObjectDeletedError:
                session._remove_newly_deleted([state])
                return None
        return instance
    else:
        return None