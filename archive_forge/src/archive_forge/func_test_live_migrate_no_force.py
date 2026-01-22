from unittest import mock
from openstack.compute.v2 import flavor
from openstack.compute.v2 import server
from openstack.image.v2 import image
from openstack.tests.unit import base
def test_live_migrate_no_force(self):
    sot = server.Server(**EXAMPLE)

    class FakeEndpointData:
        min_microversion = None
        max_microversion = None
    self.sess.get_endpoint_data.return_value = FakeEndpointData()
    ex = self.assertRaises(ValueError, sot.live_migrate, self.sess, host='HOST2', force=False, block_migration=False)
    self.assertIn("Live migration on this cloud implies 'force'", str(ex))