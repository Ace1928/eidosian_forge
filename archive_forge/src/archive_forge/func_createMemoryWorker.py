from zope.interface import implementer
from . import IWorker
from ._convenience import Quit
def createMemoryWorker():
    """
    Create an L{IWorker} that does nothing but defer work, to be performed
    later.

    @return: a worker that will enqueue work to perform later, and a callable
        that will perform one element of that work.
    @rtype: 2-L{tuple} of (L{IWorker}, L{callable})
    """

    def perform():
        if not worker._pending:
            return False
        if worker._pending[0] is NoMoreWork:
            return False
        worker._pending.pop(0)()
        return True
    worker = MemoryWorker()
    return (worker, perform)