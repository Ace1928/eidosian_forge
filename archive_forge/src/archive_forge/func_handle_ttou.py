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
def handle_ttou(self):
    """        SIGTTOU handling.
        Decreases the number of workers by one.
        """
    if self.num_workers <= 1:
        return
    self.num_workers -= 1
    self.manage_workers()