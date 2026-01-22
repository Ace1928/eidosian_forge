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
def check_max_zombies(self):
    """
        Check if we've reached max_zombie_threads_before_die; if so
        then kill the entire process.
        """
    if not self.max_zombie_threads_before_die:
        return
    found = []
    now = time.time()
    for thread_id, (time_killed, worker) in self.dying_threads.items():
        if not self.thread_exists(thread_id):
            try:
                del self.dying_threads[thread_id]
            except KeyError:
                pass
            continue
        if now - time_killed > self.dying_limit:
            found.append(thread_id)
    if found:
        self.logger.info('Found %s zombie threads', found)
    if len(found) > self.max_zombie_threads_before_die:
        self.logger.fatal('Exiting process because %s zombie threads is more than %s limit', len(found), self.max_zombie_threads_before_die)
        self.notify_problem('Exiting process because %(found)s zombie threads (more than limit of %(limit)s)\nBad threads (ids):\n  %(ids)s\n' % dict(found=len(found), limit=self.max_zombie_threads_before_die, ids='\n  '.join(map(str, found))), subject='Process restart (too many zombie threads)')
        self.shutdown(10)
        print('Shutting down', threading.current_thread())
        raise ServerExit(3)