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
def index_irange(self, table, name, start, end):
    """Return items in a named index between start/end inclusive"""
    return self.tables[table].rows.indexes[name].irange(start, end)