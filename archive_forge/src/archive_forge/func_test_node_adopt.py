from unittest import mock
from openstack.clustering.v1 import _proxy
from openstack.clustering.v1 import action
from openstack.clustering.v1 import build_info
from openstack.clustering.v1 import cluster
from openstack.clustering.v1 import cluster_attr
from openstack.clustering.v1 import cluster_policy
from openstack.clustering.v1 import event
from openstack.clustering.v1 import node
from openstack.clustering.v1 import policy
from openstack.clustering.v1 import policy_type
from openstack.clustering.v1 import profile
from openstack.clustering.v1 import profile_type
from openstack.clustering.v1 import receiver
from openstack.clustering.v1 import service
from openstack import proxy as proxy_base
from openstack.tests.unit import test_proxy_base
@mock.patch.object(proxy_base.Proxy, '_get_resource')
def test_node_adopt(self, mock_get):
    mock_node = node.Node.new()
    mock_get.return_value = mock_node
    self._verify('openstack.clustering.v1.node.Node.adopt', self.proxy.adopt_node, method_kwargs={'preview': False, 'foo': 'bar'}, expected_args=[self.proxy], expected_kwargs={'preview': False, 'foo': 'bar'})
    mock_get.assert_called_once_with(node.Node, None)