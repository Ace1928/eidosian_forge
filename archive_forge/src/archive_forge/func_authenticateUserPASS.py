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
def authenticateUserPASS(self, user, password):
    """
        Perform authentication for a username/password login.

        @type user: L{bytes}
        @param user: The name of the user attempting to log in.

        @type password: L{bytes}
        @param password: The password to authenticate with.

        @rtype: L{Deferred <defer.Deferred>} which successfully results in
            3-L{tuple} of (E{1}) L{IMailbox <pop3.IMailbox>}, (E{2}) L{IMailbox
            <pop3.IMailbox>} provider, (E{3}) no-argument callable
        @return: A deferred which fires when authentication is complete.  If
            successful, it returns a L{pop3.IMailbox} interface, a mailbox,
            and a function to be invoked with the session is terminated.
            If authentication fails, the deferred fails with an
            L{UnathorizedLogin <cred.error.UnauthorizedLogin>} error.

        @raise cred.error.UnauthorizedLogin: When authentication fails.
        """
    if self.portal is not None:
        return self.portal.login(cred.credentials.UsernamePassword(user, password), None, IMailbox)
    raise cred.error.UnauthorizedLogin()