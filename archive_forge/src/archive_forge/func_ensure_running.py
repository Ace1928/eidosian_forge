import errno
import os
import selectors
import signal
import socket
import struct
import sys
import threading
import warnings
from . import connection
from . import process
from .context import reduction
from . import resource_tracker
from . import spawn
from . import util
def ensure_running(self):
    """Make sure that a fork server is running.

        This can be called from any process.  Note that usually a child
        process will just reuse the forkserver started by its parent, so
        ensure_running() will do nothing.
        """
    with self._lock:
        resource_tracker.ensure_running()
        if self._forkserver_pid is not None:
            pid, status = os.waitpid(self._forkserver_pid, os.WNOHANG)
            if not pid:
                return
            os.close(self._forkserver_alive_fd)
            self._forkserver_address = None
            self._forkserver_alive_fd = None
            self._forkserver_pid = None
        cmd = 'from multiprocess.forkserver import main; ' + 'main(%d, %d, %r, **%r)'
        if self._preload_modules:
            desired_keys = {'main_path', 'sys_path'}
            data = spawn.get_preparation_data('ignore')
            data = {x: y for x, y in data.items() if x in desired_keys}
        else:
            data = {}
        with socket.socket(socket.AF_UNIX) as listener:
            address = connection.arbitrary_address('AF_UNIX')
            listener.bind(address)
            if not util.is_abstract_socket_namespace(address):
                os.chmod(address, 384)
            listener.listen()
            alive_r, alive_w = os.pipe()
            try:
                fds_to_pass = [listener.fileno(), alive_r]
                cmd %= (listener.fileno(), alive_r, self._preload_modules, data)
                exe = spawn.get_executable()
                args = [exe] + util._args_from_interpreter_flags()
                args += ['-c', cmd]
                pid = util.spawnv_passfds(exe, args, fds_to_pass)
            except:
                os.close(alive_w)
                raise
            finally:
                os.close(alive_r)
            self._forkserver_address = address
            self._forkserver_alive_fd = alive_w
            self._forkserver_pid = pid