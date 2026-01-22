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
def serve_client(self, conn):
    """
        Handle requests from the proxies in a particular process/thread
        """
    util.debug('starting server thread to service %r', threading.current_thread().name)
    recv = conn.recv
    send = conn.send
    id_to_obj = self.id_to_obj
    while not self.stop_event.is_set():
        try:
            methodname = obj = None
            request = recv()
            ident, methodname, args, kwds = request
            try:
                obj, exposed, gettypeid = id_to_obj[ident]
            except KeyError as ke:
                try:
                    obj, exposed, gettypeid = self.id_to_local_proxy_obj[ident]
                except KeyError:
                    raise ke
            if methodname not in exposed:
                raise AttributeError('method %r of %r object is not in exposed=%r' % (methodname, type(obj), exposed))
            function = getattr(obj, methodname)
            try:
                res = function(*args, **kwds)
            except Exception as e:
                msg = ('#ERROR', e)
            else:
                typeid = gettypeid and gettypeid.get(methodname, None)
                if typeid:
                    rident, rexposed = self.create(conn, typeid, res)
                    token = Token(typeid, self.address, rident)
                    msg = ('#PROXY', (rexposed, token))
                else:
                    msg = ('#RETURN', res)
        except AttributeError:
            if methodname is None:
                msg = ('#TRACEBACK', format_exc())
            else:
                try:
                    fallback_func = self.fallback_mapping[methodname]
                    result = fallback_func(self, conn, ident, obj, *args, **kwds)
                    msg = ('#RETURN', result)
                except Exception:
                    msg = ('#TRACEBACK', format_exc())
        except EOFError:
            util.debug('got EOF -- exiting thread serving %r', threading.current_thread().name)
            sys.exit(0)
        except Exception:
            msg = ('#TRACEBACK', format_exc())
        try:
            try:
                send(msg)
            except Exception:
                send(('#UNSERIALIZABLE', format_exc()))
        except Exception as e:
            util.info('exception in thread serving %r', threading.current_thread().name)
            util.info(' ... message was %r', msg)
            util.info(' ... exception was %r', e)
            conn.close()
            sys.exit(1)