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
def server_runner(wsgi_app, global_conf, **kwargs):
    from paste.deploy.converters import asbool
    for name in ['port', 'socket_timeout', 'threadpool_workers', 'threadpool_hung_thread_limit', 'threadpool_kill_thread_limit', 'threadpool_dying_limit', 'threadpool_spawn_if_under', 'threadpool_max_zombie_threads_before_die', 'threadpool_hung_check_period', 'threadpool_max_requests', 'request_queue_size']:
        if name in kwargs:
            kwargs[name] = int(kwargs[name])
    for name in ['use_threadpool', 'daemon_threads']:
        if name in kwargs:
            kwargs[name] = asbool(kwargs[name])
    threadpool_options = {}
    for name, value in list(kwargs.items()):
        if name.startswith('threadpool_') and name != 'threadpool_workers':
            threadpool_options[name[len('threadpool_'):]] = value
            del kwargs[name]
    if 'error_email' not in threadpool_options and 'error_email' in global_conf:
        threadpool_options['error_email'] = global_conf['error_email']
    kwargs['threadpool_options'] = threadpool_options
    serve(wsgi_app, **kwargs)