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
def popCommandQueue(self):
    """
        Return the front element of the command queue, or None if empty.
        """
    if self.actionQueue:
        return self.actionQueue.pop(0)
    else:
        return None