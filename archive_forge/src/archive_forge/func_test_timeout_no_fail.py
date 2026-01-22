from unittest import mock
from openstack.baremetal.v1 import _proxy
from openstack.baremetal.v1 import allocation
from openstack.baremetal.v1 import chassis
from openstack.baremetal.v1 import driver
from openstack.baremetal.v1 import node
from openstack.baremetal.v1 import port
from openstack.baremetal.v1 import port_group
from openstack.baremetal.v1 import volume_connector
from openstack.baremetal.v1 import volume_target
from openstack import exceptions
from openstack.tests.unit import base
from openstack.tests.unit import test_proxy_base
def test_timeout_no_fail(self, mock_get):
    mock_get.return_value._check_state_reached.return_value = False
    mock_get.return_value.id = '1234'
    result = self.proxy.wait_for_nodes_provision_state(['abcd'], 'fake state', timeout=0.001, fail=False)
    mock_get.return_value._check_state_reached.assert_called_with(self.proxy, 'fake state', True)
    self.assertEqual([], result.success)
    self.assertEqual([mock_get.return_value], result.timeout)
    self.assertEqual([], result.failure)