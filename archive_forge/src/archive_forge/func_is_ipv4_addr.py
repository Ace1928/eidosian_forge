import re
import struct
import sys
import eventlet
from eventlet import patcher
from eventlet.green import _socket_nodns
from eventlet.green import os
from eventlet.green import time
from eventlet.green import select
from eventlet.green import ssl
def is_ipv4_addr(host):
    """Return True if host is a valid IPv4 address"""
    if not isinstance(host, str):
        return False
    try:
        dns.ipv4.inet_aton(host)
    except dns.exception.SyntaxError:
        return False
    else:
        return True