from openstack.placement.v1 import _proxy
from openstack.placement.v1 import resource_class
from openstack.placement.v1 import resource_provider
from openstack.placement.v1 import resource_provider_inventory
from openstack.tests.unit import test_proxy_base as test_proxy_base
def test_resource_provider_set_aggregates(self):
    self._verify('openstack.placement.v1.resource_provider.ResourceProvider.set_aggregates', self.proxy.set_resource_provider_aggregates, method_args=['value', 'a', 'b'], expected_args=[self.proxy], expected_kwargs={'aggregates': ('a', 'b')})