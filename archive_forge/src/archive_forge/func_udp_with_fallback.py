import base64
import contextlib
import enum
import errno
import os
import os.path
import selectors
import socket
import struct
import time
from typing import Any, Dict, Optional, Tuple, Union
import dns._features
import dns.exception
import dns.inet
import dns.message
import dns.name
import dns.quic
import dns.rcode
import dns.rdataclass
import dns.rdatatype
import dns.serial
import dns.transaction
import dns.tsig
import dns.xfr
def udp_with_fallback(q: dns.message.Message, where: str, timeout: Optional[float]=None, port: int=53, source: Optional[str]=None, source_port: int=0, ignore_unexpected: bool=False, one_rr_per_rrset: bool=False, ignore_trailing: bool=False, udp_sock: Optional[Any]=None, tcp_sock: Optional[Any]=None, ignore_errors: bool=False) -> Tuple[dns.message.Message, bool]:
    """Return the response to the query, trying UDP first and falling back
    to TCP if UDP results in a truncated response.

    *q*, a ``dns.message.Message``, the query to send

    *where*, a ``str`` containing an IPv4 or IPv6 address,  where to send the message.

    *timeout*, a ``float`` or ``None``, the number of seconds to wait before the query
    times out.  If ``None``, the default, wait forever.

    *port*, an ``int``, the port send the message to.  The default is 53.

    *source*, a ``str`` containing an IPv4 or IPv6 address, specifying the source
    address.  The default is the wildcard address.

    *source_port*, an ``int``, the port from which to send the message. The default is
    0.

    *ignore_unexpected*, a ``bool``.  If ``True``, ignore responses from unexpected
    sources.

    *one_rr_per_rrset*, a ``bool``.  If ``True``, put each RR into its own RRset.

    *ignore_trailing*, a ``bool``.  If ``True``, ignore trailing junk at end of the
    received message.

    *udp_sock*, a ``socket.socket``, or ``None``, the socket to use for the UDP query.
    If ``None``, the default, a socket is created.  Note that if a socket is provided,
    it must be a nonblocking datagram socket, and the *source* and *source_port* are
    ignored for the UDP query.

    *tcp_sock*, a ``socket.socket``, or ``None``, the connected socket to use for the
    TCP query.  If ``None``, the default, a socket is created.  Note that if a socket is
    provided, it must be a nonblocking connected stream socket, and *where*, *source*
    and *source_port* are ignored for the TCP query.

    *ignore_errors*, a ``bool``.  If various format errors or response mismatches occur
    while listening for UDP, ignore them and keep listening for a valid response. The
    default is ``False``.

    Returns a (``dns.message.Message``, tcp) tuple where tcp is ``True`` if and only if
    TCP was used.
    """
    try:
        response = udp(q, where, timeout, port, source, source_port, ignore_unexpected, one_rr_per_rrset, ignore_trailing, True, udp_sock, ignore_errors)
        return (response, False)
    except dns.message.Truncated:
        response = tcp(q, where, timeout, port, source, source_port, one_rr_per_rrset, ignore_trailing, tcp_sock)
        return (response, True)