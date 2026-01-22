from unittest import mock
from openstack.clustering.v1 import cluster
from openstack.tests.unit import base
def test_policy_update(self):
    sot = cluster.Cluster(**FAKE)
    resp = mock.Mock()
    resp.json = mock.Mock(return_value='')
    sess = mock.Mock()
    sess.post = mock.Mock(return_value=resp)
    params = {'enabled': False}
    self.assertEqual('', sot.policy_update(sess, 'POLICY', **params))
    url = 'clusters/%s/actions' % sot.id
    body = {'policy_update': {'policy_id': 'POLICY', 'enabled': False}}
    sess.post.assert_called_once_with(url, json=body)