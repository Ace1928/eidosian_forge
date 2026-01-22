from __future__ import annotations
from itertools import chain
from itertools import groupby
from itertools import zip_longest
import operator
from . import attributes
from . import exc as orm_exc
from . import loading
from . import sync
from .base import state_str
from .. import exc as sa_exc
from .. import future
from .. import sql
from .. import util
from ..engine import cursor as _cursor
from ..sql import operators
from ..sql.elements import BooleanClauseList
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
def delete_obj(base_mapper, states, uowtransaction):
    """Issue ``DELETE`` statements for a list of objects.

    This is called within the context of a UOWTransaction during a
    flush operation.

    """
    states_to_delete = list(_organize_states_for_delete(base_mapper, states, uowtransaction))
    table_to_mapper = base_mapper._sorted_tables
    for table in reversed(list(table_to_mapper.keys())):
        mapper = table_to_mapper[table]
        if table not in mapper._pks_by_table:
            continue
        elif mapper.inherits and mapper.passive_deletes:
            continue
        delete = _collect_delete_commands(base_mapper, uowtransaction, table, states_to_delete)
        _emit_delete_statements(base_mapper, uowtransaction, mapper, table, delete)
    for state, state_dict, mapper, connection, update_version_id in states_to_delete:
        mapper.dispatch.after_delete(mapper, connection, state)