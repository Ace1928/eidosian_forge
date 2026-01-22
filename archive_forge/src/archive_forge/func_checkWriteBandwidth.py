import sys
from typing import Optional, Type
from zope.interface import directlyProvides, providedBy
from twisted.internet import error, interfaces
from twisted.internet.interfaces import ILoggingContext
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.python import log
def checkWriteBandwidth(self):
    if self.writtenThisSecond > self.writeLimit:
        self.throttleWrites()
        throttleTime = float(self.writtenThisSecond) / self.writeLimit - 1.0
        self.unthrottleWritesID = self.callLater(throttleTime, self.unthrottleWrites)
    self.writtenThisSecond = 0
    self.checkWriteBandwidthID = self.callLater(1, self.checkWriteBandwidth)