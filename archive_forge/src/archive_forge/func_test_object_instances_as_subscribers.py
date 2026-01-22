from unittest import mock
import ddt
from oslo_db import exception as db_exc
from oslotest import base
from neutron_lib.callbacks import events
from neutron_lib.callbacks import exceptions
from neutron_lib.callbacks import manager
from neutron_lib.callbacks import priority_group
from neutron_lib.callbacks import resources
def test_object_instances_as_subscribers(self):
    """Ensures that the manager doesn't think these are equivalent."""
    a = GloriousObjectWithCallback()
    b = ObjectWithCallback()
    c = ObjectWithCallback()
    for o in (a, b, c):
        self.manager.subscribe(o.callback, resources.PORT, events.BEFORE_CREATE)
        self.manager.subscribe(o.callback, resources.PORT, events.BEFORE_CREATE)
    self.manager.publish(resources.PORT, events.BEFORE_CREATE, mock.ANY, payload=events.EventPayload(object()))
    self.assertEqual(1, a.counter)
    self.assertEqual(1, b.counter)
    self.assertEqual(1, c.counter)