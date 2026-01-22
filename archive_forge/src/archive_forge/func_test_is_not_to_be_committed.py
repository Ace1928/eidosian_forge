from unittest import mock
from neutron_lib.callbacks import events
from oslotest import base
def test_is_not_to_be_committed(self):
    e = events.DBEventPayload(mock.ANY, states=['s1'], resource_id='1a')
    self.assertFalse(e.is_to_be_committed)