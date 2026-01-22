from unittest import mock
from neutron_lib.callbacks import events
from oslotest import base
def test_latest_state_with_desired_state(self):
    desired_state = object()
    e = events.DBEventPayload(mock.ANY, states=[object()], desired_state=desired_state)
    self.assertEqual(desired_state, e.latest_state)