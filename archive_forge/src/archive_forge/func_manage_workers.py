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
def manage_workers(self):
    """        Maintain the number of workers by spawning or killing
        as required.
        """
    if len(self.WORKERS) < self.num_workers:
        self.spawn_workers()
    workers = self.WORKERS.items()
    workers = sorted(workers, key=lambda w: w[1].age)
    while len(workers) > self.num_workers:
        pid, _ = workers.pop(0)
        self.kill_worker(pid, signal.SIGTERM)
    active_worker_count = len(workers)
    if self._last_logged_active_worker_count != active_worker_count:
        self._last_logged_active_worker_count = active_worker_count
        self.log.debug('{0} workers'.format(active_worker_count), extra={'metric': 'gunicorn.workers', 'value': active_worker_count, 'mtype': 'gauge'})