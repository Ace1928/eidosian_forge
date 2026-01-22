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
def cooperative_yield(self):
    """Hook for cooperatively yielding to eventlet/gevent/asyncio/etc.

        When a block of code is going to spend a lot of time cpu-bound without
        doing any I/O, it can cause greenthread/coroutine libraries to block.
        This call should be added to code where this can happen, but defaults
        to doing nothing to avoid overhead where it is not needed.
        """