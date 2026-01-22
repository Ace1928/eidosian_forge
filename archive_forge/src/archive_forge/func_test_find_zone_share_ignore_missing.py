import uuid
from openstack import exceptions
from openstack.tests.functional import base
from openstack import utils
def test_find_zone_share_ignore_missing(self):
    zone_share = self.operator_cloud.dns.find_zone_share(self.zone, 'bogus_id')
    self.assertIsNone(zone_share)