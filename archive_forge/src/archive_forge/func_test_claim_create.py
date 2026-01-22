from unittest import mock
from openstack.message.v2 import _proxy
from openstack.message.v2 import claim
from openstack.message.v2 import message
from openstack.message.v2 import queue
from openstack.message.v2 import subscription
from openstack import proxy as proxy_base
from openstack.tests.unit import test_proxy_base
def test_claim_create(self):
    self._verify('openstack.message.v2.claim.Claim.create', self.proxy.create_claim, method_args=['test_queue'], expected_args=[self.proxy], expected_kwargs={'base_path': None})