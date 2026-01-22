from __future__ import annotations
import warnings
from binascii import hexlify
from functools import lru_cache
from hashlib import md5
from typing import Dict
from zope.interface import Interface, implementer
from OpenSSL import SSL, crypto
from OpenSSL._util import lib as pyOpenSSLlib
import attr
from constantly import FlagConstant, Flags, NamedConstant, Names
from incremental import Version
from twisted.internet.abstract import isIPAddress, isIPv6Address
from twisted.internet.defer import Deferred
from twisted.internet.error import CertificateError, VerifyError
from twisted.internet.interfaces import (
from twisted.python import log, util
from twisted.python.compat import nativeString
from twisted.python.deprecate import _mutuallyExclusiveArguments, deprecated
from twisted.python.failure import Failure
from twisted.python.randbytes import secureRandom
from ._idna import _idnaBytes
def protocolNegotiationMechanisms():
    """
    Checks whether your versions of PyOpenSSL and OpenSSL are recent enough to
    support protocol negotiation, and if they are, what kind of protocol
    negotiation is supported.

    @return: A combination of flags from L{ProtocolNegotiationSupport} that
        indicate which mechanisms for protocol negotiation are supported.
    @rtype: L{constantly.FlagConstant}
    """
    support = ProtocolNegotiationSupport.NOSUPPORT
    ctx = SSL.Context(SSL.SSLv23_METHOD)
    try:
        ctx.set_npn_advertise_callback(lambda c: None)
    except (AttributeError, NotImplementedError):
        pass
    else:
        support |= ProtocolNegotiationSupport.NPN
    try:
        ctx.set_alpn_select_callback(lambda c: None)
    except (AttributeError, NotImplementedError):
        pass
    else:
        support |= ProtocolNegotiationSupport.ALPN
    return support