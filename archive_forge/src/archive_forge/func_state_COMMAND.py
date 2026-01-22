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
def state_COMMAND(self, line):
    """
        Handle received lines for the COMMAND state in which commands from the
        client are expected.

        @type line: L{bytes}
        @param line: A received command.
        """
    try:
        return self.processCommand(*line.split(b' '))
    except (ValueError, AttributeError, POP3Error, TypeError) as e:
        log.err()
        self.failResponse(b': '.join([b'bad protocol or server', e.__class__.__name__.encode('utf-8'), b''.join(e.args)]))