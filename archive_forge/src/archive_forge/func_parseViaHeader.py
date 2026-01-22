import socket
import time
import warnings
from collections import OrderedDict
from typing import Dict, List
from zope.interface import Interface, implementer
from twisted import cred
from twisted.internet import defer, protocol, reactor
from twisted.protocols import basic
from twisted.python import log
def parseViaHeader(value):
    """
    Parse a Via header.

    @return: The parsed version of this header.
    @rtype: L{Via}
    """
    parts = value.split(';')
    sent, params = (parts[0], parts[1:])
    protocolinfo, by = sent.split(' ', 1)
    by = by.strip()
    result = {}
    pname, pversion, transport = protocolinfo.split('/')
    if pname != 'SIP' or pversion != '2.0':
        raise ValueError(f'wrong protocol or version: {value!r}')
    result['transport'] = transport
    if ':' in by:
        host, port = by.split(':')
        result['port'] = int(port)
        result['host'] = host
    else:
        result['host'] = by
    for p in params:
        p = p.strip().split(' ', 1)
        if len(p) == 1:
            p, comment = (p[0], '')
        else:
            p, comment = p
        if p == 'hidden':
            result['hidden'] = True
            continue
        parts = p.split('=', 1)
        if len(parts) == 1:
            name, value = (parts[0], None)
        else:
            name, value = parts
            if name in ('rport', 'ttl'):
                value = int(value)
        result[name] = value
    return Via(**result)