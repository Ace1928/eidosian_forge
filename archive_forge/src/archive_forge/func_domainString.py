from __future__ import annotations
import inspect
import random
import socket
import struct
from io import BytesIO
from itertools import chain
from typing import Optional, Sequence, SupportsInt, Union, overload
from zope.interface import Attribute, Interface, implementer
from twisted.internet import defer, protocol
from twisted.internet.error import CannotListenError
from twisted.python import failure, log, randbytes, util as tputil
from twisted.python.compat import cmp, comparable, nativeString
from twisted.names.error import (
def domainString(domain: str | bytes) -> bytes:
    """
    Coerce a domain name string to bytes.

    L{twisted.names} represents domain names as L{bytes}, but many interfaces
    accept L{bytes} or a text string (L{unicode} on Python 2, L{str} on Python
    3). This function coerces text strings using IDNA encoding --- see
    L{encodings.idna}.

    Note that DNS is I{case insensitive} but I{case preserving}. This function
    doesn't normalize case, so you'll still need to do that whenever comparing
    the strings it returns.

    @param domain: A domain name.  If passed as a text string it will be
        C{idna} encoded.
    @type domain: L{bytes} or L{str}

    @returns: L{bytes} suitable for network transmission.
    @rtype: L{bytes}

    @since: Twisted 20.3.0
    """
    if isinstance(domain, str):
        domain = domain.encode('idna')
    if not isinstance(domain, bytes):
        raise TypeError('Expected {} or {} but found {!r} of type {}'.format(bytes.__name__, str.__name__, domain, type(domain)))
    return domain