from unittest import mock
from openstack.clustering.v1 import node
from openstack.tests.unit import base
def test_adopt(self):
    sot = node.Node.new()
    resp = mock.Mock()
    resp.headers = {}
    resp.json = mock.Mock(return_value={'foo': 'bar'})
    resp.status_code = 200
    sess = mock.Mock()
    sess.post = mock.Mock(return_value=resp)
    res = sot.adopt(sess, False, param='value')
    self.assertEqual(sot, res)
    sess.post.assert_called_once_with('nodes/adopt', json={'param': 'value'})