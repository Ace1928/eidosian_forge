from taskflow import exceptions as excp
from taskflow import states
from taskflow import test
def test_invalid_flow_states(self):
    invalids = [(states.RUNNING, states.PENDING), (states.REVERTED, states.RUNNING), (states.RESUMING, states.RUNNING)]
    for start_state, end_state in invalids:
        self.assertRaises(excp.InvalidState, states.check_flow_transition, start_state, end_state)