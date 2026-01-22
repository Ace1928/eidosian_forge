from unittest import mock
from neutron_lib.callbacks import events
from oslotest import base
def test_latest_state_with_states(self):
    body = object()
    states = [object(), object()]
    e = events.EventPayload(mock.ANY, request_body=body, states=states)
    self.assertEqual(states[-1], e.latest_state)