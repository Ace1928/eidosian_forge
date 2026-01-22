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
def decref(self, c, ident):
    if ident not in self.id_to_refcount and ident in self.id_to_local_proxy_obj:
        util.debug('Server DECREF skipping %r', ident)
        return
    with self.mutex:
        if self.id_to_refcount[ident] <= 0:
            raise AssertionError('Id {0!s} ({1!r}) has refcount {2:n}, not 1+'.format(ident, self.id_to_obj[ident], self.id_to_refcount[ident]))
        self.id_to_refcount[ident] -= 1
        if self.id_to_refcount[ident] == 0:
            del self.id_to_refcount[ident]
    if ident not in self.id_to_refcount:
        self.id_to_obj[ident] = (None, (), None)
        util.debug('disposing of obj with id %r', ident)
        with self.mutex:
            del self.id_to_obj[ident]