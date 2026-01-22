import sys
from typing import Optional, Type
from zope.interface import directlyProvides, providedBy
from twisted.internet import error, interfaces
from twisted.internet.interfaces import ILoggingContext
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.python import log
def checkReadBandwidth(self):
    """
        Checks if we've passed bandwidth limits.
        """
    if self.readThisSecond > self.readLimit:
        self.throttleReads()
        throttleTime = float(self.readThisSecond) / self.readLimit - 1.0
        self.unthrottleReadsID = self.callLater(throttleTime, self.unthrottleReads)
    self.readThisSecond = 0
    self.checkReadBandwidthID = self.callLater(1, self.checkReadBandwidth)