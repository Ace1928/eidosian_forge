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
def ftp_USER(self, username):
    """
        First part of login.  Get the username the peer wants to
        authenticate as.
        """
    if not username:
        return defer.fail(CmdSyntaxError('USER requires an argument'))
    self._user = username
    self.state = self.INAUTH
    if self.factory.allowAnonymous and self._user == self.factory.userAnonymous:
        return GUEST_NAME_OK_NEED_EMAIL
    else:
        return (USR_NAME_OK_NEED_PASS, username)