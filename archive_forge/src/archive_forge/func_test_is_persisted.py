from unittest import mock
from neutron_lib.callbacks import events
from oslotest import base
def test_is_persisted(self):
    e = events.DBEventPayload(mock.ANY, states=['s1'], resource_id='1a')
    self.assertTrue(e.is_persisted)