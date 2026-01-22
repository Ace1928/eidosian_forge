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
def generatePortCommand(self, portCmd):
    """
        (Private) Generates the text of a given PORT command.
        """
    factory = FTPDataPortFactory()
    factory.protocol = portCmd.protocol
    listener = reactor.listenTCP(0, factory)
    factory.port = listener

    def listenerFail(error, listener=listener):
        if listener.connected:
            listener.loseConnection()
        return error
    portCmd.fail = listenerFail
    host = self.transport.getHost().host
    port = listener.getHost().port
    portCmd.text = 'PORT ' + encodeHostPort(host, port)