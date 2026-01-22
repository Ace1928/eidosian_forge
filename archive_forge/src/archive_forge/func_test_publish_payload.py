from unittest import mock
from oslotest import base
import testtools
from neutron_lib.callbacks import events
from neutron_lib.callbacks import priority_group
from neutron_lib.callbacks import registry
from neutron_lib.callbacks import resources
from neutron_lib import fixture
def test_publish_payload(self):
    event_payload = events.EventPayload(mock.ANY)
    registry.publish('x', 'y', self, payload=event_payload)
    self.callback_manager.publish.assert_called_with('x', 'y', self, payload=event_payload)