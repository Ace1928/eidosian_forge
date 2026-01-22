from unittest import mock
from openstack.message.v2 import _proxy
from openstack.message.v2 import claim
from openstack.message.v2 import message
from openstack.message.v2 import queue
from openstack.message.v2 import subscription
from openstack import proxy as proxy_base
from openstack.tests.unit import test_proxy_base
def test_claim_get(self):
    self._verify('openstack.proxy.Proxy._get', self.proxy.get_claim, method_args=['test_queue', 'resource_or_id'], expected_args=[claim.Claim, 'resource_or_id'], expected_kwargs={'queue_name': 'test_queue'})
    self.verify_get_overrided(self.proxy, claim.Claim, 'openstack.message.v2.claim.Claim')