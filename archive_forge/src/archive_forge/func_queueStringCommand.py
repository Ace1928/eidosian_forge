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
def queueStringCommand(self, command, public=1):
    """
        Queues a string to be issued as an FTP command

        @param command: string of an FTP command to queue
        @param public: a flag intended for internal use by FTPClient.  Don't
            change it unless you know what you're doing.

        @return: a L{Deferred} that will be called when the response to the
            command has been received.
        """
    ftpCommand = FTPCommand(command, public)
    self.queueCommand(ftpCommand)
    return ftpCommand.deferred