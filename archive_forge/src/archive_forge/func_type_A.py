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
def type_A(self, code):
    if code == '' or code == 'N':
        self.binary = False
        return (TYPE_SET_OK, 'A' + code)
    else:
        return defer.fail(CmdArgSyntaxError(code))