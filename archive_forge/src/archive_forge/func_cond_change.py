import collections
import enum
import functools
import uuid
import ovs.db.data as data
import ovs.db.parser
import ovs.db.schema
import ovs.jsonrpc
import ovs.ovsuuid
import ovs.poller
import ovs.vlog
from ovs.db import custom_index
from ovs.db import error
def cond_change(self, table_name, cond):
    """Sets the condition for 'table_name' to 'cond', which should be a
        conditional expression suitable for use directly in the OVSDB
        protocol, with the exception that the empty condition []
        matches no rows (instead of matching every row).  That is, []
        is equivalent to [False], not to [True].
        """
    table = self.tables.get(table_name)
    if not table:
        raise error.Error('Unknown table "%s"' % table_name)
    if cond == []:
        cond = [False]
    if table.condition_state.latest != cond:
        table.condition_state.init(cond)
        self.cond_changed = True
    if table.condition_state.new:
        any_reqs = any((t.condition_state.request for t in self.tables.values()))
        return self.cond_seqno + int(any_reqs) + 1
    return self.cond_seqno + int(bool(table.condition_state.requested))