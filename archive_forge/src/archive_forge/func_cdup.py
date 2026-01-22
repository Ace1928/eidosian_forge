import errno
import fnmatch
import os
import re
import stat
import time
from zope.interface import Interface, implementer
from twisted import copyright
from twisted.cred import checkers, credentials, error as cred_error, portal
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.protocols import basic, policies
from twisted.python import failure, filepath, log
def cdup(self):
    """
        Issues the CDUP (Change Directory UP) command.

        @return: a L{Deferred} that will be called when done.
        """
    return self.queueStringCommand('CDUP')