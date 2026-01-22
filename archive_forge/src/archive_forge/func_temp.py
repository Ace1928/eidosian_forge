import sys
import threading
import signal
import array
import queue
import time
import types
import os
from os import getpid
from traceback import format_exc
from . import connection
from .context import reduction, get_spawning_popen, ProcessError
from . import pool
from . import process
from . import util
from . import get_context
def temp(self, /, *args, **kwds):
    util.debug('requesting creation of a shared %r object', typeid)
    token, exp = self._create(typeid, *args, **kwds)
    proxy = proxytype(token, self._serializer, manager=self, authkey=self._authkey, exposed=exp)
    conn = self._Client(token.address, authkey=self._authkey)
    dispatch(conn, None, 'decref', (token.id,))
    return proxy