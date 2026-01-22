import struct
from twisted.internet import defer
from twisted.protocols import basic
from twisted.python import failure, log
def invalidQuery(self):
    self.transport.loseConnection()