from unittest import mock
from openstack.clustering.v1 import cluster
from openstack.tests.unit import base
def test_policy_detach(self):
    sot = cluster.Cluster(**FAKE)
    resp = mock.Mock()
    resp.json = mock.Mock(return_value='')
    sess = mock.Mock()
    sess.post = mock.Mock(return_value=resp)
    self.assertEqual('', sot.policy_detach(sess, 'POLICY'))
    url = 'clusters/%s/actions' % sot.id
    body = {'policy_detach': {'policy_id': 'POLICY'}}
    sess.post.assert_called_once_with(url, json=body)