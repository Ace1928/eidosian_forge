from datetime import datetime
import errno
import os
import select
import socket
import ssl
import sys
from gunicorn import http
from gunicorn.http import wsgi
from gunicorn import sock
from gunicorn import util
from gunicorn.workers import base
def run_for_one(self, timeout):
    listener = self.sockets[0]
    while self.alive:
        self.notify()
        try:
            self.accept(listener)
            continue
        except EnvironmentError as e:
            if e.errno not in (errno.EAGAIN, errno.ECONNABORTED, errno.EWOULDBLOCK):
                raise
        if not self.is_parent_alive():
            return
        try:
            self.wait(timeout)
        except StopWaiting:
            return