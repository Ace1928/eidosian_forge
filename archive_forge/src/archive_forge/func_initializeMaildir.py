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
def initializeMaildir(dir):
    """
    Create a maildir user directory if it doesn't already exist.

    @type dir: L{bytes}
    @param dir: The path name for a user directory.
    """
    dir = os.fsdecode(dir)
    if not os.path.isdir(dir):
        os.mkdir(dir, 448)
        for subdir in ['new', 'cur', 'tmp', '.Trash']:
            os.mkdir(os.path.join(dir, subdir), 448)
        for subdir in ['new', 'cur', 'tmp']:
            os.mkdir(os.path.join(dir, '.Trash', subdir), 448)
        open(os.path.join(dir, '.Trash', 'maildirfolder'), 'w').close()