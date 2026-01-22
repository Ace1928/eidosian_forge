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
def ftp_SIZE(self, path):
    """
        File SIZE

        The FTP command, SIZE OF FILE (SIZE), is used to obtain the transfer
        size of a file from the server-FTP process.  This is the exact number
        of octets (8 bit bytes) that would be transmitted over the data
        connection should that file be transmitted.  This value will change
        depending on the current STRUcture, MODE, and TYPE of the data
        connection or of a data connection that would be created were one
        created now.  Thus, the result of the SIZE command is dependent on
        the currently established STRU, MODE, and TYPE parameters.

        The SIZE command returns how many octets would be transferred if the
        file were to be transferred using the current transfer structure,
        mode, and type.  This command is normally used in conjunction with
        the RESTART (REST) command when STORing a file to a remote server in
        STREAM mode, to determine the restart point.  The server-PI might
        need to read the partially transferred file, do any appropriate
        conversion, and count the number of octets that would be generated
        when sending the file in order to correctly respond to this command.
        Estimates of the file transfer size MUST NOT be returned; only
        precise information is acceptable.

        http://tools.ietf.org/html/rfc3659
        """
    try:
        newsegs = toSegments(self.workingDirectory, path)
    except InvalidPath:
        return defer.fail(FileNotFoundError(path))

    def cbStat(result):
        size, = result
        return (FILE_STATUS, str(size))
    return self.shell.stat(newsegs, ('size',)).addCallback(cbStat)