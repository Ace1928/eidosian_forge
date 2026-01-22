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
def trustRootFromCertificates(certificates):
    """
    Builds an object that trusts multiple root L{Certificate}s.

    When passed to L{optionsForClientTLS}, connections using those options will
    reject any server certificate not signed by at least one of the
    certificates in the `certificates` list.

    @since: 16.0

    @param certificates: All certificates which will be trusted.
    @type certificates: C{iterable} of L{CertBase}

    @rtype: L{IOpenSSLTrustRoot}
    @return: an object suitable for use as the trustRoot= keyword argument to
        L{optionsForClientTLS}
    """
    certs = []
    for cert in certificates:
        if isinstance(cert, CertBase):
            cert = cert.original
        else:
            raise TypeError('certificates items must be twisted.internet.ssl.CertBase instances')
        certs.append(cert)
    return OpenSSLCertificateAuthorities(certs)