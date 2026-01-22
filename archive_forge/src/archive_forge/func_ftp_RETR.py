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
def ftp_RETR(self, path):
    """
        This command causes the content of a file to be sent over the data
        transfer channel. If the path is to a folder, an error will be raised.

        @type path: C{str}
        @param path: The path to the file which should be transferred over the
        data transfer channel.

        @rtype: L{Deferred}
        @return: a L{Deferred} which will be fired when the transfer is done.
        """
    if self.dtpInstance is None:
        raise BadCmdSequenceError('PORT or PASV required before RETR')
    try:
        newsegs = toSegments(self.workingDirectory, path)
    except InvalidPath:
        return defer.fail(FileNotFoundError(path))
    self.setTimeout(None)

    def enableTimeout(result):
        self.setTimeout(self.factory.timeOut)
        return result
    if not self.binary:
        cons = ASCIIConsumerWrapper(self.dtpInstance)
    else:
        cons = self.dtpInstance

    def cbSent(result):
        return (TXFR_COMPLETE_OK,)

    def ebSent(err):
        log.msg('Unexpected error attempting to transmit file to client:')
        log.err(err)
        if err.check(FTPCmdError):
            return err
        return (CNX_CLOSED_TXFR_ABORTED,)

    def cbOpened(file):
        if self.dtpInstance.isConnected:
            self.reply(DATA_CNX_ALREADY_OPEN_START_XFR)
        else:
            self.reply(FILE_STATUS_OK_OPEN_DATA_CNX)
        d = file.send(cons)
        d.addCallbacks(cbSent, ebSent)
        return d

    def ebOpened(err):
        if not err.check(PermissionDeniedError, FileNotFoundError, IsADirectoryError):
            log.msg('Unexpected error attempting to open file for transmission:')
            log.err(err)
        if err.check(FTPCmdError):
            return (err.value.errorCode, '/'.join(newsegs))
        return (FILE_NOT_FOUND, '/'.join(newsegs))
    d = self.shell.openForReading(newsegs)
    d.addCallbacks(cbOpened, ebOpened)
    d.addBoth(enableTimeout)
    return d