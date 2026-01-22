import base64
import binascii
import warnings
from hashlib import md5
from typing import Optional
from zope.interface import implementer
from twisted import cred
from twisted.internet import defer, interfaces, task
from twisted.mail import smtp
from twisted.mail._except import POP3ClientError, POP3Error, _POP3MessageDeleted
from twisted.mail.interfaces import (
from twisted.protocols import basic, policies
from twisted.python import log
from twisted.mail._except import (
from twisted.mail._pop3client import POP3Client as AdvancedPOP3Client
def do_AUTH(self, args=None):
    """
        Handle an AUTH command.

        If the AUTH extension is not supported, send an error response.  If an
        authentication mechanism was not specified in the command, send a list
        of all supported authentication methods.  Otherwise, send an
        authentication challenge to the client and transition to the
        AUTH state.

        @type args: L{bytes} or L{None}
        @param args: The name of an authentication mechanism.
        """
    if not getattr(self.factory, 'challengers', None):
        self.failResponse(b'AUTH extension unsupported')
        return
    if args is None:
        self.successResponse('Supported authentication methods:')
        for a in self.factory.challengers:
            self.sendLine(a.upper())
        self.sendLine(b'.')
        return
    auth = self.factory.challengers.get(args.strip().upper())
    if not self.portal or not auth:
        self.failResponse(b'Unsupported SASL selected')
        return
    self._auth = auth()
    chal = self._auth.getChallenge()
    self.sendLine(b'+ ' + base64.b64encode(chal))
    self.state = 'AUTH'