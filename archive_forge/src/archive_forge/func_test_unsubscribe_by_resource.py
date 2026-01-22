from unittest import mock
from oslotest import base
import testtools
from neutron_lib.callbacks import events
from neutron_lib.callbacks import priority_group
from neutron_lib.callbacks import registry
from neutron_lib.callbacks import resources
from neutron_lib import fixture
def test_unsubscribe_by_resource(self):
    registry.unsubscribe_by_resource(my_callback, 'my-resource')
    self.callback_manager.unsubscribe_by_resource.assert_called_with(my_callback, 'my-resource')