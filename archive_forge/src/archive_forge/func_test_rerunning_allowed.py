from taskflow import exceptions as exc
from taskflow import states
from taskflow import test
def test_rerunning_allowed(self):
    self.assertTransitionAllowed(states.SUCCESS, states.RUNNING)