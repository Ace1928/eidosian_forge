import socket
import string
import struct
import time
from twisted.internet import defer, protocol, reactor
from twisted.python import log
def connectClass(self, host, port, klass, *args):
    return protocol.ClientCreator(reactor, klass, *args).connectTCP(host, port)