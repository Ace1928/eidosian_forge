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
def test_success_no_fail(self, mock_get):
    nodes = [mock.Mock(spec=node.Node, id=str(i)) for i in range(3)]
    for i, n in enumerate(nodes):
        n._check_state_reached.return_value = not i % 2
        mock_get.side_effect = nodes
    result = self.proxy.wait_for_nodes_provision_state(['abcd', node.Node(id='1234')], 'fake state', fail=False)
    self.assertEqual([nodes[0], nodes[2]], result.success)
    self.assertEqual([], result.failure)
    self.assertEqual([], result.timeout)
    for n in nodes:
        n._check_state_reached.assert_called_once_with(self.proxy, 'fake state', True)