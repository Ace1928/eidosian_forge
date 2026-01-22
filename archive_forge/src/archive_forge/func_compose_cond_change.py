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
def compose_cond_change(self):
    if not self.cond_changed:
        return
    change_requests = {}
    for table in self.tables.values():
        if table.condition_state.new is not None:
            change_requests[table.name] = [{'where': table.condition_state.new}]
            table.condition_state.request()
    if not change_requests:
        return
    self.cond_changed = False
    old_uuid = str(self.uuid)
    self.uuid = uuid.uuid1()
    params = [old_uuid, str(self.uuid), change_requests]
    return ovs.jsonrpc.Message.create_request('monitor_cond_change', params)