from unittest import mock
from openstack.block_storage.v3 import service
from openstack.tests.unit import base
@mock.patch('openstack.utils.supports_microversion', autospec=True, return_value=True)
def test_failover__with_cluster(self, mock_supports):
    self.sess.default_microversion = '3.26'
    sot = service.Service(**EXAMPLE)
    res = sot.failover(self.sess, cluster='foo', backend_id='bar')
    self.assertIsNotNone(res)
    url = 'os-services/failover'
    body = {'host': 'devstack', 'cluster': 'foo', 'backend_id': 'bar'}
    self.sess.put.assert_called_with(url, json=body, microversion='3.26')