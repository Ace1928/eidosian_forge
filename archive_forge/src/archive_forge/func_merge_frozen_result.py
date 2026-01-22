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
@util.preload_module('sqlalchemy.orm.context')
def merge_frozen_result(session, statement, frozen_result, load=True):
    """Merge a :class:`_engine.FrozenResult` back into a :class:`_orm.Session`,
    returning a new :class:`_engine.Result` object with :term:`persistent`
    objects.

    See the section :ref:`do_orm_execute_re_executing` for an example.

    .. seealso::

        :ref:`do_orm_execute_re_executing`

        :meth:`_engine.Result.freeze`

        :class:`_engine.FrozenResult`

    """
    querycontext = util.preloaded.orm_context
    if load:
        session._autoflush()
    ctx = querycontext.ORMSelectCompileState._create_entities_collection(statement, legacy=False)
    autoflush = session.autoflush
    try:
        session.autoflush = False
        mapped_entities = [i for i, e in enumerate(ctx._entities) if isinstance(e, querycontext._MapperEntity)]
        keys = [ent._label_name for ent in ctx._entities]
        keyed_tuple = result_tuple(keys, [ent._extra_entities for ent in ctx._entities])
        result = []
        for newrow in frozen_result.rewrite_rows():
            for i in mapped_entities:
                if newrow[i] is not None:
                    newrow[i] = session._merge(attributes.instance_state(newrow[i]), attributes.instance_dict(newrow[i]), load=load, _recursive={}, _resolve_conflict_map={})
            result.append(keyed_tuple(newrow))
        return frozen_result.with_new_rows(result)
    finally:
        session.autoflush = autoflush