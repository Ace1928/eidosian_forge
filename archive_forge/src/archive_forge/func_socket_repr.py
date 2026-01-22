import errno
import os
import sys
import time
import traceback
import types
import urllib.parse
import warnings
import eventlet
from eventlet import greenio
from eventlet import support
from eventlet.corolocal import local
from eventlet.green import BaseHTTPServer
from eventlet.green import socket
def socket_repr(sock):
    scheme = 'http'
    if hasattr(sock, 'do_handshake'):
        scheme = 'https'
    name = sock.getsockname()
    if sock.family == socket.AF_INET:
        hier_part = '//{}:{}'.format(*name)
    elif sock.family == socket.AF_INET6:
        hier_part = '//[{}]:{}'.format(*name[:2])
    elif sock.family == socket.AF_UNIX:
        hier_part = name
    else:
        hier_part = repr(name)
    return scheme + ':' + hier_part