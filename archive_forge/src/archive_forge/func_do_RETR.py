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
def do_RETR(self, i):
    """
        Handle a RETR command.

        @type i: L{bytes}
        @param i: A 1-based message index.

        @rtype: L{Deferred}
        @return: A deferred which triggers after the response to the RETR
            command has been issued.
        """
    return self._sendMessageContent(i, lambda fp: fp, lambda size: '%d' % (size,))