import io
import os
import signal
import sys
import time
import traceback
from datetime import datetime
from random import randint
from ssl import SSLError
from gunicorn import util
from gunicorn.http.errors import (
from gunicorn.http.wsgi import Response, default_environ
from gunicorn.reloader import reloader_engines
from gunicorn.workers.workertmp import WorkerTmp
def init_signals(self):
    for s in self.SIGNALS:
        signal.signal(s, signal.SIG_DFL)
    signal.signal(signal.SIGQUIT, self.handle_quit)
    signal.signal(signal.SIGTERM, self.handle_exit)
    signal.signal(signal.SIGINT, self.handle_quit)
    signal.signal(signal.SIGWINCH, self.handle_winch)
    signal.signal(signal.SIGUSR1, self.handle_usr1)
    signal.signal(signal.SIGABRT, self.handle_abort)
    signal.siginterrupt(signal.SIGTERM, False)
    signal.siginterrupt(signal.SIGUSR1, False)
    if hasattr(signal, 'set_wakeup_fd'):
        signal.set_wakeup_fd(self.PIPE[1])