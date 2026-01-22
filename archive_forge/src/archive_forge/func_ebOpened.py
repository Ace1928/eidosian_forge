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
def ebOpened(err):
    """
            Called when failed to open the file for reading.

            For known errors, return the FTP error code.
            For all other, return a file not found error.
            """
    if isinstance(err.value, FTPCmdError):
        return (err.value.errorCode, '/'.join(newsegs))
    log.err(err, 'Unexpected error received while opening file:')
    return (FILE_NOT_FOUND, '/'.join(newsegs))