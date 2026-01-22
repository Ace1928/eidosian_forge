import random
from openstack import connection
from openstack import exceptions
from openstack.tests.functional import base
from openstack import utils
def test_create_rs(self):
    zone = self.conn.dns.get_zone(self.zone)
    self.assertIsNotNone(self.conn.dns.create_recordset(zone=zone, name='www.{zone}'.format(zone=zone.name), type='A', description='Example zone rec', ttl=3600, records=['192.168.1.1']))