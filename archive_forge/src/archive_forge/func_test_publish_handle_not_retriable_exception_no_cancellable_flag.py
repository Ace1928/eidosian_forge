from unittest import mock
import ddt
from oslo_db import exception as db_exc
from oslotest import base
from neutron_lib.callbacks import events
from neutron_lib.callbacks import exceptions
from neutron_lib.callbacks import manager
from neutron_lib.callbacks import priority_group
from neutron_lib.callbacks import resources
def test_publish_handle_not_retriable_exception_no_cancellable_flag(self):
    self.manager.subscribe(callback_raise_not_retriable, resources.PORT, events.AFTER_INIT)
    self.manager.publish(resources.PORT, events.AFTER_INIT, self, payload=self.event_payload)