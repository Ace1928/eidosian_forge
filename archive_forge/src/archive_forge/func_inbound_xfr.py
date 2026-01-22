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
def inbound_xfr(where: str, txn_manager: dns.transaction.TransactionManager, query: Optional[dns.message.Message]=None, port: int=53, timeout: Optional[float]=None, lifetime: Optional[float]=None, source: Optional[str]=None, source_port: int=0, udp_mode: UDPMode=UDPMode.NEVER) -> None:
    """Conduct an inbound transfer and apply it via a transaction from the
    txn_manager.

    *where*, a ``str`` containing an IPv4 or IPv6 address,  where
    to send the message.

    *txn_manager*, a ``dns.transaction.TransactionManager``, the txn_manager
    for this transfer (typically a ``dns.zone.Zone``).

    *query*, the query to send.  If not supplied, a default query is
    constructed using information from the *txn_manager*.

    *port*, an ``int``, the port send the message to.  The default is 53.

    *timeout*, a ``float``, the number of seconds to wait for each
    response message.  If None, the default, wait forever.

    *lifetime*, a ``float``, the total number of seconds to spend
    doing the transfer.  If ``None``, the default, then there is no
    limit on the time the transfer may take.

    *source*, a ``str`` containing an IPv4 or IPv6 address, specifying
    the source address.  The default is the wildcard address.

    *source_port*, an ``int``, the port from which to send the message.
    The default is 0.

    *udp_mode*, a ``dns.query.UDPMode``, determines how UDP is used
    for IXFRs.  The default is ``dns.UDPMode.NEVER``, i.e. only use
    TCP.  Other possibilities are ``dns.UDPMode.TRY_FIRST``, which
    means "try UDP but fallback to TCP if needed", and
    ``dns.UDPMode.ONLY``, which means "try UDP and raise
    ``dns.xfr.UseTCP`` if it does not succeed.

    Raises on errors.
    """
    if query is None:
        query, serial = dns.xfr.make_query(txn_manager)
    else:
        serial = dns.xfr.extract_serial_from_query(query)
    rdtype = query.question[0].rdtype
    is_ixfr = rdtype == dns.rdatatype.IXFR
    origin = txn_manager.from_wire_origin()
    wire = query.to_wire()
    af, destination, source = _destination_and_source(where, port, source, source_port)
    _, expiration = _compute_times(lifetime)
    retry = True
    while retry:
        retry = False
        if is_ixfr and udp_mode != UDPMode.NEVER:
            sock_type = socket.SOCK_DGRAM
            is_udp = True
        else:
            sock_type = socket.SOCK_STREAM
            is_udp = False
        with _make_socket(af, sock_type, source) as s:
            _connect(s, destination, expiration)
            if is_udp:
                _udp_send(s, wire, None, expiration)
            else:
                tcpmsg = struct.pack('!H', len(wire)) + wire
                _net_write(s, tcpmsg, expiration)
            with dns.xfr.Inbound(txn_manager, rdtype, serial, is_udp) as inbound:
                done = False
                tsig_ctx = None
                while not done:
                    _, mexpiration = _compute_times(timeout)
                    if mexpiration is None or (expiration is not None and mexpiration > expiration):
                        mexpiration = expiration
                    if is_udp:
                        rwire, _ = _udp_recv(s, 65535, mexpiration)
                    else:
                        ldata = _net_read(s, 2, mexpiration)
                        l, = struct.unpack('!H', ldata)
                        rwire = _net_read(s, l, mexpiration)
                    r = dns.message.from_wire(rwire, keyring=query.keyring, request_mac=query.mac, xfr=True, origin=origin, tsig_ctx=tsig_ctx, multi=not is_udp, one_rr_per_rrset=is_ixfr)
                    try:
                        done = inbound.process_message(r)
                    except dns.xfr.UseTCP:
                        assert is_udp
                        if udp_mode == UDPMode.ONLY:
                            raise
                        done = True
                        retry = True
                        udp_mode = UDPMode.NEVER
                        continue
                    tsig_ctx = r.tsig_ctx
                if not retry and query.keyring and (not r.had_tsig):
                    raise dns.exception.FormError('missing TSIG')