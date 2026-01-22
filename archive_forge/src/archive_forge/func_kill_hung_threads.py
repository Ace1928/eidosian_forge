import atexit
import traceback
import io
import socket, sys, threading
import posixpath
import time
import os
from itertools import count
import _thread
import queue
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import unquote, urlsplit
from paste.util import converters
import logging
def kill_hung_threads(self):
    """
        Tries to kill any hung threads
        """
    if not self.kill_thread_limit:
        return
    now = time.time()
    max_time = 0
    total_time = 0
    idle_workers = 0
    starting_workers = 0
    working_workers = 0
    killed_workers = 0
    for worker in self.workers:
        if not hasattr(worker, 'thread_id'):
            starting_workers += 1
            continue
        time_started, info = self.worker_tracker.get(worker.thread_id, (None, None))
        if time_started is None:
            idle_workers += 1
            continue
        working_workers += 1
        max_time = max(max_time, now - time_started)
        total_time += now - time_started
        if now - time_started > self.kill_thread_limit:
            self.logger.warning('Thread %s hung (working on task for %i seconds)', worker.thread_id, now - time_started)
            try:
                import pprint
                info_desc = pprint.pformat(info)
            except:
                out = io.StringIO()
                traceback.print_exc(file=out)
                info_desc = 'Error:\n%s' % out.getvalue()
            self.notify_problem('Killing worker thread (id=%(thread_id)s) because it has been \nworking on task for %(time)s seconds (limit is %(limit)s)\nInfo on task:\n%(info)s' % dict(thread_id=worker.thread_id, time=now - time_started, limit=self.kill_thread_limit, info=info_desc))
            self.kill_worker(worker.thread_id)
            killed_workers += 1
    if working_workers:
        ave_time = float(total_time) / working_workers
        ave_time = '%.2fsec' % ave_time
    else:
        ave_time = 'N/A'
    self.logger.info('kill_hung_threads status: %s threads (%s working, %s idle, %s starting) ave time %s, max time %.2fsec, killed %s workers' % (idle_workers + starting_workers + working_workers, working_workers, idle_workers, starting_workers, ave_time, max_time, killed_workers))
    self.check_max_zombies()