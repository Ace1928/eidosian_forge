from unittest import mock
from openstack.clustering.v1 import node
from openstack.tests.unit import base
def test_adopt_preview(self):
    sot = node.Node.new()
    resp = mock.Mock()
    resp.headers = {}
    resp.json = mock.Mock(return_value={'foo': 'bar'})
    sess = mock.Mock()
    sess.post = mock.Mock(return_value=resp)
    attrs = {'identity': 'fake-resource-id', 'overrides': {}, 'type': 'os.nova.server-1.0', 'snapshot': False}
    res = sot.adopt(sess, True, **attrs)
    self.assertEqual({'foo': 'bar'}, res)
    sess.post.assert_called_once_with('nodes/adopt-preview', json=attrs)