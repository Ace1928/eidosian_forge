from unittest import mock
from openstack.clustering.v1 import cluster
from openstack.tests.unit import base
def test_scale_in(self):
    sot = cluster.Cluster(**FAKE)
    resp = mock.Mock()
    resp.json = mock.Mock(return_value='')
    sess = mock.Mock()
    sess.post = mock.Mock(return_value=resp)
    self.assertEqual('', sot.scale_in(sess, 3))
    url = 'clusters/%s/actions' % sot.id
    body = {'scale_in': {'count': 3}}
    sess.post.assert_called_once_with(url, json=body)