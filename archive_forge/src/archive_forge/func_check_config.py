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
@classmethod
def check_config(cls, cfg, log):
    max_keepalived = cfg.worker_connections - cfg.threads
    if max_keepalived <= 0 and cfg.keepalive:
        log.warning('No keepalived connections can be handled. ' + 'Check the number of worker connections and threads.')