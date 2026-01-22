import socket
import string
import struct
import time
from twisted.internet import defer, protocol, reactor
from twisted.python import log
def listenClass(self, port, klass, *args):
    serv = reactor.listenTCP(port, klass(*args))
    return defer.succeed(serv.getHost()[1:])