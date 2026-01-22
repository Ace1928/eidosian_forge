from twisted.python.components import proxyForInterface
from twisted.python.context import call, get
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase
from .. import AlreadyQuit, IWorker, Team, createMemoryWorker
def test_quitQuits(self):
    """
        L{Team.quit} causes all idle workers, as well as the coordinator
        worker, to quit.
        """
    for x in range(10):
        self.team.do(list)
    self.performAllOutstandingWork()
    self.team.quit()
    self.performAllOutstandingWork()
    self.assertEqual(len(self.allUnquitWorkers), 0)
    self.assertRaises(AlreadyQuit, self.coordinator.quit)