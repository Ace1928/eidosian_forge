from unittest import mock
from openstack.compute.v2 import flavor
from openstack.compute.v2 import server
from openstack.image.v2 import image
from openstack.tests.unit import base
def test_live_migrate_no_microversion_force_true(self):
    sot = server.Server(**EXAMPLE)

    class FakeEndpointData:
        min_microversion = None
        max_microversion = None
    self.sess.get_endpoint_data.return_value = FakeEndpointData()
    res = sot.live_migrate(self.sess, host='HOST2', force=True, block_migration=True, disk_over_commit=True)
    self.assertIsNone(res)
    url = 'servers/IDENTIFIER/action'
    body = {'os-migrateLive': {'host': 'HOST2', 'disk_over_commit': True, 'block_migration': True}}
    headers = {'Accept': ''}
    self.sess.post.assert_called_with(url, json=body, headers=headers, microversion=self.sess.default_microversion)