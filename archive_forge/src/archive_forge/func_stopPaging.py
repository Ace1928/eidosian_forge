from zope.interface import implementer
from twisted.internet import defer, interfaces
from twisted.protocols import basic
from twisted.python.failure import Failure
from twisted.spread import pb
def stopPaging(self):
    """
        Call this when you're done paging.
        """
    self._stillPaging = 0