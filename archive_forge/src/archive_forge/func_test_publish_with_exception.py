from unittest import mock
import ddt
from oslo_db import exception as db_exc
from oslotest import base
from neutron_lib.callbacks import events
from neutron_lib.callbacks import exceptions
from neutron_lib.callbacks import manager
from neutron_lib.callbacks import priority_group
from neutron_lib.callbacks import resources
def test_publish_with_exception(self):
    with mock.patch.object(self.manager, '_notify_loop') as n:
        n.return_value = ['error']
        self.assertRaises(exceptions.CallbackFailure, self.manager.publish, mock.ANY, events.BEFORE_CREATE, mock.ANY, payload=self.event_payload)
        expected_calls = [mock.call(mock.ANY, 'before_create', mock.ANY, self.event_payload), mock.call(mock.ANY, 'abort_create', mock.ANY, self.event_payload)]
        n.assert_has_calls(expected_calls)