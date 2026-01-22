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
def processCommand(self, cmd, *params):

    def call_ftp_command(command):
        method = getattr(self, 'ftp_' + command, None)
        if method is not None:
            return method(*params)
        return defer.fail(CmdNotImplementedError(command))
    cmd = cmd.upper()
    if cmd in self.PUBLIC_COMMANDS:
        return call_ftp_command(cmd)
    elif self.state == self.UNAUTH:
        if cmd == 'USER':
            return self.ftp_USER(*params)
        elif cmd == 'PASS':
            return (BAD_CMD_SEQ, 'USER required before PASS')
        else:
            return NOT_LOGGED_IN
    elif self.state == self.INAUTH:
        if cmd == 'PASS':
            return self.ftp_PASS(*params)
        else:
            return (BAD_CMD_SEQ, 'PASS required after USER')
    elif self.state == self.AUTHED:
        return call_ftp_command(cmd)
    elif self.state == self.RENAMING:
        if cmd == 'RNTO':
            return self.ftp_RNTO(*params)
        else:
            return (BAD_CMD_SEQ, 'RNTO required after RNFR')