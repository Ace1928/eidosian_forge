from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import allocation
from openstack import exceptions
from openstack.tests.unit import base
def test_already_active(self, mock_fetch):
    self.allocation.state = 'active'
    allocation = self.allocation.wait(None)
    self.assertIs(allocation, self.allocation)
    self.assertFalse(mock_fetch.called)