from unittest import mock
from oslotest import base
import testtools
from neutron_lib.callbacks import events
from neutron_lib.callbacks import priority_group
from neutron_lib.callbacks import registry
from neutron_lib.callbacks import resources
from neutron_lib import fixture
def test_unsubscribe_all(self):
    registry.unsubscribe_all(my_callback)
    self.callback_manager.unsubscribe_all.assert_called_with(my_callback)