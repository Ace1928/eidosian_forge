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
def formatUIDListResponse(msgs, getUidl):
    """
    Format a list of message sizes into a complete UIDL response.

    This generator function is intended to be used with
    L{Cooperator <twisted.internet.task.Cooperator>}.

    @type msgs: L{list} of L{int}
    @param msgs: A list of message sizes.

    @type getUidl: one-argument callable returning bytes
    @param getUidl: A callable which takes a message index number and returns
        the UID of the corresponding message in the mailbox.

    @rtype: L{bytes}
    @return: Yields a series of strings which make up a complete UIDL response.
    """
    yield successResponse('')
    yield from formatUIDListLines(msgs, getUidl)
    yield b'.\r\n'