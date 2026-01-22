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
def ftp_RNTO(self, toName):
    fromName = self._fromName
    del self._fromName
    self.state = self.AUTHED
    try:
        fromsegs = toSegments(self.workingDirectory, fromName)
        tosegs = toSegments(self.workingDirectory, toName)
    except InvalidPath:
        return defer.fail(FileNotFoundError(fromName))
    return self.shell.rename(fromsegs, tosegs).addCallback(lambda ign: (REQ_FILE_ACTN_COMPLETED_OK,))