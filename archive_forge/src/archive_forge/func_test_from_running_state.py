from taskflow import exceptions as exc
from taskflow import states
from taskflow import test
def test_from_running_state(self):
    self.assertTransitions(from_state=states.RUNNING, allowed=(states.SUCCESS, states.FAILURE), ignored=(states.REVERTING, states.RUNNING, states.PENDING, states.REVERTED))