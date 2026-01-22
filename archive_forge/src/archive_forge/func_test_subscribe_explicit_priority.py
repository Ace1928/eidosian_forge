from unittest import mock
from oslotest import base
import testtools
from neutron_lib.callbacks import events
from neutron_lib.callbacks import priority_group
from neutron_lib.callbacks import registry
from neutron_lib.callbacks import resources
from neutron_lib import fixture
def test_subscribe_explicit_priority(self):
    registry.subscribe(my_callback, 'my-resource', 'my-event', PRI_CALLBACK)
    self.callback_manager.subscribe.assert_called_with(my_callback, 'my-resource', 'my-event', PRI_CALLBACK, False)