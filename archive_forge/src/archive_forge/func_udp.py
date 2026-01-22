from __future__ import generators
import errno
import select
import socket
import struct
import sys
import time
import dns.exception
import dns.inet
import dns.name
import dns.message
import dns.rcode
import dns.rdataclass
import dns.rdatatype
from ._compat import long, string_types, PY3
def udp(q, where, timeout=None, port=53, af=None, source=None, source_port=0, ignore_unexpected=False, one_rr_per_rrset=False, ignore_trailing=False):
    """Return the response obtained after sending a query via UDP.

    *q*, a ``dns.message.Message``, the query to send

    *where*, a ``text`` containing an IPv4 or IPv6 address,  where
    to send the message.

    *timeout*, a ``float`` or ``None``, the number of seconds to wait before the
    query times out.  If ``None``, the default, wait forever.

    *port*, an ``int``, the port send the message to.  The default is 53.

    *af*, an ``int``, the address family to use.  The default is ``None``,
    which causes the address family to use to be inferred from the form of
    *where*.  If the inference attempt fails, AF_INET is used.  This
    parameter is historical; you need never set it.

    *source*, a ``text`` containing an IPv4 or IPv6 address, specifying
    the source address.  The default is the wildcard address.

    *source_port*, an ``int``, the port from which to send the message.
    The default is 0.

    *ignore_unexpected*, a ``bool``.  If ``True``, ignore responses from
    unexpected sources.

    *one_rr_per_rrset*, a ``bool``.  If ``True``, put each RR into its own
    RRset.

    *ignore_trailing*, a ``bool``.  If ``True``, ignore trailing
    junk at end of the received message.

    Returns a ``dns.message.Message``.
    """
    wire = q.to_wire()
    af, destination, source = _destination_and_source(af, where, port, source, source_port)
    s = socket_factory(af, socket.SOCK_DGRAM, 0)
    received_time = None
    sent_time = None
    try:
        expiration = _compute_expiration(timeout)
        s.setblocking(0)
        if source is not None:
            s.bind(source)
        _, sent_time = send_udp(s, wire, destination, expiration)
        r, received_time = receive_udp(s, destination, expiration, ignore_unexpected, one_rr_per_rrset, q.keyring, q.mac, ignore_trailing)
    finally:
        if sent_time is None or received_time is None:
            response_time = 0
        else:
            response_time = received_time - sent_time
        s.close()
    r.time = response_time
    if not q.is_response(r):
        raise BadResponse
    return r