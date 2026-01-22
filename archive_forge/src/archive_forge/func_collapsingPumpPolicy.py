import tempfile
from zope.interface import implementer
from twisted.internet import defer, interfaces, main, protocol
from twisted.internet.interfaces import IAddress
from twisted.internet.task import deferLater
from twisted.protocols import policies
from twisted.python import failure
def collapsingPumpPolicy(queue, target):
    """
    L{collapsingPumpPolicy} is a policy which collapses all outstanding chunks
    into a single string and delivers it to the target.

    @see: L{loopbackAsync}
    """
    bytes = []
    while queue:
        chunk = queue.get()
        if chunk is None:
            break
        bytes.append(chunk)
    if bytes:
        target.dataReceived(b''.join(bytes))