from taskflow import exceptions as exc
from taskflow import states
from taskflow import test
def test_from_pending_state(self):
    self.assertTransitions(from_state=states.PENDING, allowed=(states.RUNNING,), ignored=(states.PENDING, states.REVERTING, states.SUCCESS, states.FAILURE, states.REVERTED))