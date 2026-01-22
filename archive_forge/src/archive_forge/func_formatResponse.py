import base64
from binascii import hexlify
from hashlib import md5, sha1
from zope.interface.verify import verifyObject
from twisted.cred.credentials import (
from twisted.cred.error import LoginFailed
from twisted.internet.address import IPv4Address
from twisted.python.compat import networkString
from twisted.trial.unittest import TestCase
def formatResponse(self, quotes=True, **kw):
    """
        Format all given keyword arguments and their values suitably for use as
        the value of an HTTP header.

        @types quotes: C{bool}
        @param quotes: A flag indicating whether to quote the values of each
            field in the response.

        @param **kw: Keywords and C{bytes} values which will be treated as field
            name/value pairs to include in the result.

        @rtype: C{bytes}
        @return: The given fields formatted for use as an HTTP header value.
        """
    if 'username' not in kw:
        kw['username'] = self.username
    if 'realm' not in kw:
        kw['realm'] = self.realm
    if 'algorithm' not in kw:
        kw['algorithm'] = self.algorithm
    if 'qop' not in kw:
        kw['qop'] = self.qop
    if 'cnonce' not in kw:
        kw['cnonce'] = self.cnonce
    if 'uri' not in kw:
        kw['uri'] = self.uri
    if quotes:
        quote = b'"'
    else:
        quote = b''
    return b', '.join([b''.join((networkString(k), b'=', quote, v, quote)) for k, v in kw.items() if v is not None])