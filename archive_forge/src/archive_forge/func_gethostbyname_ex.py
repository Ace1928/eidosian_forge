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
def gethostbyname_ex(hostname):
    """Replacement for Python's socket.gethostbyname_ex"""
    if is_ipv4_addr(hostname):
        return (hostname, [], [hostname])
    ans = resolve(hostname)
    aliases = getaliases(hostname)
    addrs = [rr.address for rr in ans.rrset]
    qname = str(ans.qname)
    if qname[-1] == '.':
        qname = qname[:-1]
    return (qname, aliases, addrs)