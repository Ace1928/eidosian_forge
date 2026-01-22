from unittest import mock
from openstack.clustering.v1 import cluster
from openstack.tests.unit import base
def test_replace_nodes(self):
    sot = cluster.Cluster(**FAKE)
    resp = mock.Mock()
    resp.json = mock.Mock(return_value='')
    sess = mock.Mock()
    sess.post = mock.Mock(return_value=resp)
    self.assertEqual('', sot.replace_nodes(sess, {'node-22': 'node-44'}))
    url = 'clusters/%s/actions' % sot.id
    body = {'replace_nodes': {'nodes': {'node-22': 'node-44'}}}
    sess.post.assert_called_once_with(url, json=body)