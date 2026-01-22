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
def storeFile(self, path, offset=0):
    """
        Store a file at the given path.

        This method issues the 'STOR' FTP command.

        @return: A tuple of two L{Deferred}s:
                  - L{Deferred} L{IFinishableConsumer}. You must call
                    the C{finish} method on the IFinishableConsumer when the
                    file is completely transferred.
                  - L{Deferred} list of control-connection responses.
        """
    cmds = ['STOR ' + self.escapePath(path)]
    if offset:
        cmds.insert(0, 'REST ' + str(offset))
    return self.sendToConnection(cmds)