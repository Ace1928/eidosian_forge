from unittest import mock
from oslotest import base
import testtools
from neutron_lib.callbacks import events
from neutron_lib.callbacks import priority_group
from neutron_lib.callbacks import registry
from neutron_lib.callbacks import resources
from neutron_lib import fixture
def test_subscribe(self):
    registry.subscribe(my_callback, 'my-resource', 'my-event')
    self.callback_manager.subscribe.assert_called_with(my_callback, 'my-resource', 'my-event', priority_group.PRIORITY_DEFAULT, False)