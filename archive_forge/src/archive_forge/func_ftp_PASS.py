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
def ftp_PASS(self, password):
    """
        Second part of login.  Get the password the peer wants to
        authenticate with.
        """
    if self.factory.allowAnonymous and self._user == self.factory.userAnonymous:
        creds = credentials.Anonymous()
        reply = GUEST_LOGGED_IN_PROCEED
    else:
        creds = credentials.UsernamePassword(self._user, password)
        reply = USR_LOGGED_IN_PROCEED
    del self._user

    def _cbLogin(result):
        interface, avatar, logout = result
        assert interface is IFTPShell, 'The realm is busted, jerk.'
        self.shell = avatar
        self.logout = logout
        self.workingDirectory = []
        self.state = self.AUTHED
        return reply

    def _ebLogin(failure):
        failure.trap(cred_error.UnauthorizedLogin, cred_error.UnhandledCredentials)
        self.state = self.UNAUTH
        raise AuthorizationError
    d = self.portal.login(creds, None, IFTPShell)
    d.addCallbacks(_cbLogin, _ebLogin)
    return d