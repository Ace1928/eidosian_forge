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
def incref(self, c, ident):
    with self.mutex:
        try:
            self.id_to_refcount[ident] += 1
        except KeyError as ke:
            if ident in self.id_to_local_proxy_obj:
                self.id_to_refcount[ident] = 1
                self.id_to_obj[ident] = self.id_to_local_proxy_obj[ident]
                obj, exposed, gettypeid = self.id_to_obj[ident]
                util.debug('Server re-enabled tracking & INCREF %r', ident)
            else:
                raise ke