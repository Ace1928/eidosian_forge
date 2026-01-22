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
def queueLogin(self, username, password):
    """
        Login: send the username, send the password, and
        set retrieval mode to binary
        """
    FTPClientBasic.queueLogin(self, username, password)
    d = self.queueStringCommand('TYPE I', public=0)
    d.addErrback(self.fail)
    d.addErrback(lambda x: None)