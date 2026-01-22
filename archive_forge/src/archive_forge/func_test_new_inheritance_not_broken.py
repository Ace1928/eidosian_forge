from unittest import mock
from oslotest import base
import testtools
from neutron_lib.callbacks import events
from neutron_lib.callbacks import priority_group
from neutron_lib.callbacks import registry
from neutron_lib.callbacks import resources
from neutron_lib import fixture
def test_new_inheritance_not_broken(self):
    self.assertTrue(AnotherObjectWithDecoratedCallback().new_called)