from __future__ import annotations
from zope.interface import implementedBy, implementer, implementer_only
from OpenSSL import SSL
from twisted.internet import interfaces, tcp
from twisted.internet._sslverify import (
def cacheContext(self):
    if self._context is None:
        ctx = self._contextFactory(self.sslmethod)
        ctx.set_options(SSL.OP_NO_SSLv2)
        ctx.use_certificate_file(self.certificateFileName)
        ctx.use_privatekey_file(self.privateKeyFileName)
        self._context = ctx