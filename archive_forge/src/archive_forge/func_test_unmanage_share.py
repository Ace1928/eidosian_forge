from unittest import mock
from keystoneauth1 import adapter
from openstack.shared_file_system.v2 import share
from openstack.tests.unit import base
def test_unmanage_share(self):
    sot = share.Share(**EXAMPLE)
    microversion = sot._get_microversion(self.sess, action='patch')
    self.assertIsNone(sot.unmanage(self.sess))
    url = 'shares/%s/action' % IDENTIFIER
    body = {'unmanage': None}
    self.sess.post.assert_called_with(url, json=body, headers={'Accept': ''}, microversion=microversion)