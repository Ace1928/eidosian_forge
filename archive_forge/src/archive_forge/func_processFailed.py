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
def processFailed(err):
    if err.check(FTPCmdError):
        self.sendLine(err.value.response())
    elif err.check(TypeError) and any((msg in err.value.args[0] for msg in ('takes exactly', 'required positional argument'))):
        self.reply(SYNTAX_ERR, f'{cmd} requires an argument.')
    else:
        log.msg('Unexpected FTP error')
        log.err(err)
        self.reply(REQ_ACTN_NOT_TAKEN, 'internal server error')