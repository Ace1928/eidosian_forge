from zope.interface import implementer
from twisted.internet import defer, interfaces
from twisted.protocols import basic
from twisted.python.failure import Failure
from twisted.spread import pb
def sendNextPage(self):
    """
        Get the first chunk read and send it to collector.
        """
    if not self.chunks:
        return
    val = self.chunks.pop(0)
    self.producer.resumeProducing()
    self.collector.callRemote('gotPage', val, pbanswer=False)