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
def doPassive(response):
    """Connect to the port specified in the response to PASV"""
    host, port = decodeHostPort(response[-1][4:])
    f = _PassiveConnectionFactory(protocol)
    _mutable[0] = self.connectFactory(host, port, f)