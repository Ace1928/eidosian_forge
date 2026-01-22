import errno
import os
import re
import socket
import ssl
from contextlib import contextmanager
from ssl import SSLError
from struct import pack, unpack
from .exceptions import UnexpectedFrame
from .platform import KNOWN_TCP_OPTS, SOL_TCP
from .utils import set_cloexec
def to_host_port(host, default=AMQP_PORT):
    """Convert hostname:port string to host, port tuple."""
    port = default
    m = IPV6_LITERAL.match(host)
    if m:
        host = m.group(1)
        if m.group(2):
            port = int(m.group(2))
    elif ':' in host:
        host, port = host.rsplit(':', 1)
        port = int(port)
    return (host, port)