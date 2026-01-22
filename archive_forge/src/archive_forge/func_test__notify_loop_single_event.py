from unittest import mock
import ddt
from oslo_db import exception as db_exc
from oslotest import base
from neutron_lib.callbacks import events
from neutron_lib.callbacks import exceptions
from neutron_lib.callbacks import manager
from neutron_lib.callbacks import priority_group
from neutron_lib.callbacks import resources
def test__notify_loop_single_event(self):
    self.manager.subscribe(callback_1, resources.PORT, events.BEFORE_CREATE)
    self.manager.subscribe(callback_2, resources.PORT, events.BEFORE_CREATE)
    self.manager._notify_loop(resources.PORT, events.BEFORE_CREATE, mock.ANY, payload=mock.ANY)
    self.assertEqual(1, callback_1.counter)
    self.assertEqual(1, callback_2.counter)