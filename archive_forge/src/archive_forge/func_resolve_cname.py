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
def resolve_cname(host):
    """Return the canonical name of a hostname"""
    try:
        ans = resolver.query(host, dns.rdatatype.CNAME)
    except dns.resolver.NoAnswer:
        return host
    except dns.exception.Timeout:
        _raise_new_error(EAI_EAGAIN_ERROR)
    except dns.exception.DNSException:
        _raise_new_error(EAI_NODATA_ERROR)
    else:
        return str(ans[0].target)