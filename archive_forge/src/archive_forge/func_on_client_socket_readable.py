from concurrent import futures
import errno
import os
import selectors
import socket
import ssl
import sys
import time
from collections import deque
from datetime import datetime
from functools import partial
from threading import RLock
from . import base
from .. import http
from .. import util
from .. import sock
from ..http import wsgi
def on_client_socket_readable(self, conn, client):
    with self._lock:
        self.poller.unregister(client)
        if conn.initialized:
            try:
                self._keep.remove(conn)
            except ValueError:
                return
    self.enqueue_req(conn)