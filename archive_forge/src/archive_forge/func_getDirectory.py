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
def getDirectory(self):
    """
        Returns the current remote directory.

        @return: a L{Deferred} that will be called back with a C{str} giving
            the remote directory or which will errback with L{CommandFailed}
            if an error response is returned.
        """

    def cbParse(result):
        try:
            if int(result[0].split(' ', 1)[0]) != 257:
                raise ValueError
        except (IndexError, ValueError):
            return failure.Failure(CommandFailed(result))
        path = parsePWDResponse(result[0])
        if path is None:
            return failure.Failure(CommandFailed(result))
        return path
    return self.pwd().addCallback(cbParse)