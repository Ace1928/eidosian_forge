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
def reexec(self):
    """        Relaunch the master and workers.
        """
    if self.reexec_pid != 0:
        self.log.warning('USR2 signal ignored. Child exists.')
        return
    if self.master_pid != 0:
        self.log.warning('USR2 signal ignored. Parent exists.')
        return
    master_pid = os.getpid()
    self.reexec_pid = os.fork()
    if self.reexec_pid != 0:
        return
    self.cfg.pre_exec(self)
    environ = self.cfg.env_orig.copy()
    environ['GUNICORN_PID'] = str(master_pid)
    if self.systemd:
        environ['LISTEN_PID'] = str(os.getpid())
        environ['LISTEN_FDS'] = str(len(self.LISTENERS))
    else:
        environ['GUNICORN_FD'] = ','.join((str(lnr.fileno()) for lnr in self.LISTENERS))
    os.chdir(self.START_CTX['cwd'])
    os.execvpe(self.START_CTX[0], self.START_CTX['args'], environ)