import errno
import os
import random
import select
import signal
import sys
import time
import traceback
from gunicorn.errors import HaltServer, AppImportError
from gunicorn.pidfile import Pidfile
from gunicorn import sock, systemd, util
from gunicorn import __version__, SERVER_SOFTWARE
def kill_worker(self, pid, sig):
    """        Kill a worker

        :attr pid: int, worker pid
        :attr sig: `signal.SIG*` value
         """
    try:
        os.kill(pid, sig)
    except OSError as e:
        if e.errno == errno.ESRCH:
            try:
                worker = self.WORKERS.pop(pid)
                worker.tmp.close()
                self.cfg.worker_exit(self, worker)
                return
            except (KeyError, OSError):
                return
        raise