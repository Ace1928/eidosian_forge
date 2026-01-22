from unittest import mock
import ddt
from oslo_db import exception as db_exc
from oslotest import base
from neutron_lib.callbacks import events
from neutron_lib.callbacks import exceptions
from neutron_lib.callbacks import manager
from neutron_lib.callbacks import priority_group
from neutron_lib.callbacks import resources
@mock.patch('neutron_lib.callbacks.manager.LOG')
def test_callback_order(self, _logger):
    self.manager.subscribe(callback_1, 'my-resource', 'my-event', PRI_MED)
    self.manager.subscribe(callback_2, 'my-resource', 'my-event', PRI_HIGH)
    self.manager.subscribe(callback_3, 'my-resource', 'my-event', PRI_LOW)
    self.assertEqual(3, len(self.manager._callbacks['my-resource']['my-event']))
    self.manager.unsubscribe(callback_3, 'my-resource', 'my-event')
    self.manager.publish('my-resource', 'my-event', mock.ANY, payload=self.event_payload)
    self.assertEqual(2, len(self.manager._callbacks['my-resource']['my-event']))
    self.assertEqual(0, callback_3.counter)
    self.assertEqual(1, callback_2.counter)
    self.assertEqual(1, callback_1.counter)
    callback_ids = _logger.debug.mock_calls[4][1][1]
    self.assertEqual(callback_id_2, callback_ids[0])
    self.assertEqual(callback_id_1, callback_ids[1])