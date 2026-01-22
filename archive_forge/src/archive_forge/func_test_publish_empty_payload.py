from unittest import mock
import ddt
from oslo_db import exception as db_exc
from oslotest import base
from neutron_lib.callbacks import events
from neutron_lib.callbacks import exceptions
from neutron_lib.callbacks import manager
from neutron_lib.callbacks import priority_group
from neutron_lib.callbacks import resources
def test_publish_empty_payload(self):
    notify_payload = []

    def _memo(resource, event, trigger, payload=None):
        notify_payload.append(payload)
    self.manager.subscribe(_memo, 'x', 'y')
    self.manager.publish('x', 'y', self)
    self.assertIsNone(notify_payload[0])