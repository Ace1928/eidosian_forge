from unittest import mock
import ddt
from oslo_db import exception as db_exc
from oslotest import base
from neutron_lib.callbacks import events
from neutron_lib.callbacks import exceptions
from neutron_lib.callbacks import manager
from neutron_lib.callbacks import priority_group
from neutron_lib.callbacks import resources
def test_subscribe_is_idempotent(self):
    for cancellable in (True, False):
        self.manager.subscribe(callback_1, resources.PORT, events.BEFORE_CREATE, cancellable=cancellable)
        self.manager.subscribe(callback_1, resources.PORT, events.BEFORE_CREATE, cancellable=cancellable)
    self.assertEqual(1, len(self.manager._callbacks[resources.PORT][events.BEFORE_CREATE]))
    self.assertTrue(self.manager._callbacks[resources.PORT][events.BEFORE_CREATE][0][2])
    callbacks = self.manager._index[callback_id_1][resources.PORT]
    self.assertEqual(1, len(callbacks))