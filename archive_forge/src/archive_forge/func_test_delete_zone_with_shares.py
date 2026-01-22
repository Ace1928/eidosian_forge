import random
from openstack import connection
from openstack import exceptions
from openstack.tests.functional import base
from openstack import utils
def test_delete_zone_with_shares(self):
    if not utils.supports_version(self.conn.dns, '2.1'):
        self.skipTest('Designate API version does not support shared zones.')
    zone_name = 'example-{0}.org.'.format(random.randint(1, 10000))
    zone = self.conn.dns.create_zone(name=zone_name, email='joe@example.org', type='PRIMARY', ttl=7200, description='example zone')
    self.addCleanup(self.conn.dns.delete_zone, zone)
    demo_project_id = self.operator_cloud.get_project('demo')['id']
    zone_share = self.conn.dns.create_zone_share(zone, target_project_id=demo_project_id)
    self.addCleanup(self.conn.dns.delete_zone_share, zone, zone_share)
    self.assertRaises(exceptions.BadRequestException, self.conn.dns.delete_zone, zone)
    self.conn.dns.delete_zone(zone, delete_shares=True)