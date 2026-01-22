import sys
import warnings
from functools import wraps
from io import BytesIO
from twisted.internet import defer, protocol
from twisted.python import failure
def errReceivedIsBad(self, text):
    if self.deferred is not None:
        self.onProcessEnded = defer.Deferred()
        err = _UnexpectedErrorOutput(text, self.onProcessEnded)
        self.deferred.errback(failure.Failure(err))
        self.deferred = None
        self.transport.loseConnection()