from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def dieNext(self):
    """
        Make the next result from my worker iterator be raising an
        L{UnhandledException}.
        """

    def ignoreUnhandled(failure):
        failure.trap(UnhandledException)
        return None
    self._doDieNext = True