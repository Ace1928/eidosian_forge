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
def do_APOP(self, user, digest):
    """
        Handle an APOP command.

        Perform APOP authentication and complete the authorization process with
        the L{_cbMailbox} callback function on success or the L{_ebMailbox}
        and L{_ebUnexpected} errback functions on failure.

        @type user: L{bytes}
        @param user: A username.

        @type digest: L{bytes}
        @param digest: An MD5 digest string.
        """
    d = defer.maybeDeferred(self.authenticateUserAPOP, user, digest)
    d.addCallbacks(self._cbMailbox, self._ebMailbox, callbackArgs=(user,)).addErrback(self._ebUnexpected)