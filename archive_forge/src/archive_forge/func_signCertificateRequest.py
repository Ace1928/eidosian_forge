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
def signCertificateRequest(self, issuerDistinguishedName, requestData, verifyDNCallback, serialNumber, requestFormat=crypto.FILETYPE_ASN1, certificateFormat=crypto.FILETYPE_ASN1, secondsToExpiry=60 * 60 * 24 * 365, digestAlgorithm='sha256'):
    """
        Given a blob of certificate request data and a certificate authority's
        DistinguishedName, return a blob of signed certificate data.

        If verifyDNCallback returns a Deferred, I will return a Deferred which
        fires the data when that Deferred has completed.
        """
    hlreq = CertificateRequest.load(requestData, requestFormat)
    dn = hlreq.getSubject()
    vval = verifyDNCallback(dn)

    def verified(value):
        if not value:
            raise VerifyError('DN callback {!r} rejected request DN {!r}'.format(verifyDNCallback, dn))
        return self.signRequestObject(issuerDistinguishedName, hlreq, serialNumber, secondsToExpiry, digestAlgorithm).dump(certificateFormat)
    if isinstance(vval, Deferred):
        return vval.addCallback(verified)
    else:
        return verified(vval)