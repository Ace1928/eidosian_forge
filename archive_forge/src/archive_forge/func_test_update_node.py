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
@mock.patch.object(node.Node, 'commit', autospec=True)
def test_update_node(self, mock_commit):
    self.proxy.update_node('uuid', instance_id='new value')
    mock_commit.assert_called_once_with(mock.ANY, self.proxy, retry_on_conflict=True)
    self.assertEqual('new value', mock_commit.call_args[0][0].instance_id)