from unittest import mock
import ddt
from oslo_db import exception as db_exc
from oslotest import base
from neutron_lib.callbacks import events
from neutron_lib.callbacks import exceptions
from neutron_lib.callbacks import manager
from neutron_lib.callbacks import priority_group
from neutron_lib.callbacks import resources
def test_subscribe_unknown(self):
    self.manager.subscribe(callback_1, 'my_resource', 'my-event')
    self.assertIsNotNone(self.manager._callbacks['my_resource']['my-event'])
    self.assertIn(callback_id_1, self.manager._index)