from unittest import mock
from openstack.message.v2 import _proxy
from openstack.message.v2 import claim
from openstack.message.v2 import message
from openstack.message.v2 import queue
from openstack.message.v2 import subscription
from openstack import proxy as proxy_base
from openstack.tests.unit import test_proxy_base
@mock.patch.object(proxy_base.Proxy, '_get_resource')
def test_subscription_delete_ignore(self, mock_get_resource):
    mock_get_resource.return_value = 'test_subscription'
    self.verify_delete(self.proxy.delete_subscription, subscription.Subscription, ignore_missing=True, method_args=['test_queue', 'resource_or_id'], expected_args=['test_subscription'])
    mock_get_resource.assert_called_once_with(subscription.Subscription, 'resource_or_id', queue_name='test_queue')