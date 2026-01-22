import io
import os
import socket
import stat
from hashlib import md5
from typing import IO
from zope.interface import implementer
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, interfaces, reactor
from twisted.mail import mail, pop3, smtp
from twisted.persisted import dirdbm
from twisted.protocols import basic
from twisted.python import failure, log
def startUp(self):
    """
        Start transferring the message to the mailbox.
        """
    self.createTempFile()
    if self.fh != -1:
        self.filesender = basic.FileSender()
        self.filesender.beginFileTransfer(self.msg, self)