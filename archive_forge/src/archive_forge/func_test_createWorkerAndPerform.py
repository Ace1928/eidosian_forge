from zope.interface.verify import verifyObject
from twisted.trial.unittest import SynchronousTestCase
from .. import AlreadyQuit, IWorker, createMemoryWorker
def test_createWorkerAndPerform(self) -> None:
    """
        L{createMemoryWorker} creates an L{IWorker} and a callable that can
        perform work on it.  The performer returns C{True} if it accomplished
        useful work.
        """
    worker, performer = createMemoryWorker()
    verifyObject(IWorker, worker)
    done = []
    worker.do(lambda: done.append(3))
    worker.do(lambda: done.append(4))
    self.assertEqual(done, [])
    self.assertEqual(performer(), True)
    self.assertEqual(done, [3])
    self.assertEqual(performer(), True)
    self.assertEqual(done, [3, 4])