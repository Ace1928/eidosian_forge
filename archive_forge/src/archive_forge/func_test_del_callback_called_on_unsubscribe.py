from unittest import mock
import ddt
from oslo_db import exception as db_exc
from oslotest import base
from neutron_lib.callbacks import events
from neutron_lib.callbacks import exceptions
from neutron_lib.callbacks import manager
from neutron_lib.callbacks import priority_group
from neutron_lib.callbacks import resources
@mock.patch('neutron_lib.callbacks.manager.CallbacksManager._del_callback')
def test_del_callback_called_on_unsubscribe(self, mock_cb):
    self.manager.subscribe(callback_1, 'my-resource', 'my-event')
    callback_id = self.manager._find(callback_1)
    callbacks = self.manager._callbacks['my-resource']['my-event']
    self.assertEqual(1, len(callbacks))
    self.manager.unsubscribe(callback_1, 'my-resource', 'my-event')
    mock_cb.assert_called_once_with(callbacks, callback_id)