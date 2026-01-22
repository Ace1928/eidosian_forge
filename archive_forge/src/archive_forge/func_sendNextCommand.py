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
def sendNextCommand(self):
    """
        (Private) Processes the next command in the queue.
        """
    ftpCommand = self.popCommandQueue()
    if ftpCommand is None:
        self.nextDeferred = None
        return
    if not ftpCommand.ready:
        self.actionQueue.insert(0, ftpCommand)
        reactor.callLater(1.0, self.sendNextCommand)
        self.nextDeferred = None
        return
    if ftpCommand.text == 'PORT':
        self.generatePortCommand(ftpCommand)
    if self.debug:
        log.msg('<-- %s' % ftpCommand.text)
    self.nextDeferred = ftpCommand.deferred
    self.sendLine(ftpCommand.text)