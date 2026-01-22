from unittest import mock
import ddt
from oslo_db import exception as db_exc
from oslotest import base
from neutron_lib.callbacks import events
from neutron_lib.callbacks import exceptions
from neutron_lib.callbacks import manager
from neutron_lib.callbacks import priority_group
from neutron_lib.callbacks import resources
def test_publish_none(self):
    self.manager.publish(resources.PORT, events.BEFORE_CREATE, mock.ANY, payload=self.event_payload)
    self.assertEqual(0, callback_1.counter)
    self.assertEqual(0, callback_2.counter)