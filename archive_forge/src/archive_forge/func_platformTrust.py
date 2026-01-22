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
def platformTrust():
    """
    Attempt to discover a set of trusted certificate authority certificates
    (or, in other words: trust roots, or root certificates) whose trust is
    managed and updated by tools outside of Twisted.

    If you are writing any client-side TLS code with Twisted, you should use
    this as the C{trustRoot} argument to L{CertificateOptions
    <twisted.internet.ssl.CertificateOptions>}.

    The result of this function should be like the up-to-date list of
    certificates in a web browser.  When developing code that uses
    C{platformTrust}, you can think of it that way.  However, the choice of
    which certificate authorities to trust is never Twisted's responsibility.
    Unless you're writing a very unusual application or library, it's not your
    code's responsibility either.  The user may use platform-specific tools for
    defining which server certificates should be trusted by programs using TLS.
    The purpose of using this API is to respect that decision as much as
    possible.

    This should be a set of trust settings most appropriate for I{client} TLS
    connections; i.e. those which need to verify a server's authenticity.  You
    should probably use this by default for any client TLS connection that you
    create.  For servers, however, client certificates are typically not
    verified; or, if they are, their verification will depend on a custom,
    application-specific certificate authority.

    @since: 14.0

    @note: Currently, L{platformTrust} depends entirely upon your OpenSSL build
        supporting a set of "L{default verify paths <OpenSSLDefaultPaths>}"
        which correspond to certificate authority trust roots.  Unfortunately,
        whether this is true of your system is both outside of Twisted's
        control and difficult (if not impossible) for Twisted to detect
        automatically.

        Nevertheless, this ought to work as desired by default on:

            - Ubuntu Linux machines with the U{ca-certificates
              <https://launchpad.net/ubuntu/+source/ca-certificates>} package
              installed,

            - macOS when using the system-installed version of OpenSSL (i.e.
              I{not} one installed via MacPorts or Homebrew),

            - any build of OpenSSL which has had certificate authority
              certificates installed into its default verify paths (by default,
              C{/usr/local/ssl/certs} if you've built your own OpenSSL), or

            - any process where the C{SSL_CERT_FILE} environment variable is
              set to the path of a file containing your desired CA certificates
              bundle.

        Hopefully soon, this API will be updated to use more sophisticated
        trust-root discovery mechanisms.  Until then, you can follow tickets in
        the Twisted tracker for progress on this implementation on U{Microsoft
        Windows <https://twistedmatrix.com/trac/ticket/6371>}, U{macOS
        <https://twistedmatrix.com/trac/ticket/6372>}, and U{a fallback for
        other platforms which do not have native trust management tools
        <https://twistedmatrix.com/trac/ticket/6934>}.

    @return: an appropriate trust settings object for your platform.
    @rtype: L{IOpenSSLTrustRoot}

    @raise NotImplementedError: if this platform is not yet supported by
        Twisted.  At present, only OpenSSL is supported.
    """
    return OpenSSLDefaultPaths()