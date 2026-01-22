from typing import TYPE_CHECKING, List
from twisted.trial.unittest import SynchronousTestCase
from .reactormixins import ReactorBuilder
def test_noTimers(self) -> None:
    """
        The loop can wake up just fine even if there are no timers in it.
        """
    r = self.buildReactor()
    stopped = []

    def doStop() -> None:
        r.stop()
        stopped.append('yes')

    def sleepThenStop() -> None:
        r.callFromThread(doStop)
    r.callLater(0, r.callInThread, sleepThenStop)
    r.run()
    self.assertEqual(stopped, ['yes'])