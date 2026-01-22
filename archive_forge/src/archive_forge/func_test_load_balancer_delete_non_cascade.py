from unittest import mock
import uuid
from openstack.load_balancer.v2 import _proxy
from openstack.load_balancer.v2 import amphora
from openstack.load_balancer.v2 import availability_zone
from openstack.load_balancer.v2 import availability_zone_profile
from openstack.load_balancer.v2 import flavor
from openstack.load_balancer.v2 import flavor_profile
from openstack.load_balancer.v2 import health_monitor
from openstack.load_balancer.v2 import l7_policy
from openstack.load_balancer.v2 import l7_rule
from openstack.load_balancer.v2 import listener
from openstack.load_balancer.v2 import load_balancer as lb
from openstack.load_balancer.v2 import member
from openstack.load_balancer.v2 import pool
from openstack.load_balancer.v2 import provider
from openstack.load_balancer.v2 import quota
from openstack import proxy as proxy_base
from openstack.tests.unit import test_proxy_base
@mock.patch.object(proxy_base.Proxy, '_get_resource')
def test_load_balancer_delete_non_cascade(self, mock_get_resource):
    fake_load_balancer = mock.Mock()
    fake_load_balancer.id = 'load_balancer_id'
    mock_get_resource.return_value = fake_load_balancer
    self._verify('openstack.proxy.Proxy._delete', self.proxy.delete_load_balancer, method_args=['resource_or_id', True, False], expected_args=[lb.LoadBalancer, fake_load_balancer], expected_kwargs={'ignore_missing': True})
    self.assertFalse(fake_load_balancer.cascade)
    mock_get_resource.assert_called_once_with(lb.LoadBalancer, 'resource_or_id')