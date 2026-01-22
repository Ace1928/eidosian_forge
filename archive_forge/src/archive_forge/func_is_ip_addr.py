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
def is_ip_addr(host):
    """Return True if host is a valid IPv4 or IPv6 address"""
    return is_ipv4_addr(host) or is_ipv6_addr(host)