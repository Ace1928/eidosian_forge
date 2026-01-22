from taskflow import exceptions as exc
from taskflow import states
from taskflow import test
def test_bad_transition_raises(self):
    self.assertTransitionForbidden(states.FAILURE, states.SUCCESS)