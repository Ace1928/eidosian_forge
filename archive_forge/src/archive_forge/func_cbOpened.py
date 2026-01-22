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
def cbOpened(file):
    """
            File was open for reading. Launch the data transfer channel via
            the file consumer.
            """
    d = file.receive()
    d.addCallback(cbConsumer)
    d.addCallback(lambda ignored: file.close())
    d.addCallbacks(cbSent, ebSent)
    return d