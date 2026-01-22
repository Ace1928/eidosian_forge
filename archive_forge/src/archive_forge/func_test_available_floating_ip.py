import pprint
import sys
from testtools import content
from openstack.cloud import meta
from openstack import exceptions
from openstack import proxy
from openstack.tests.functional import base
from openstack import utils
def test_available_floating_ip(self):
    fips_user = self.user_cloud.list_floating_ips()
    self.assertEqual(fips_user, [])
    new_fip = self.user_cloud.available_floating_ip()
    self.assertIsNotNone(new_fip)
    self.assertIn('id', new_fip)
    self.addCleanup(self.user_cloud.delete_floating_ip, new_fip.id)
    new_fips_user = self.user_cloud.list_floating_ips()
    self.assertEqual(new_fips_user, [new_fip])
    reuse_fip = self.user_cloud.available_floating_ip()
    self.assertEqual(reuse_fip.id, new_fip.id)