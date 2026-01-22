from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.datagen import random_zone_name
from designateclient.functionaltests.v2.fixtures import ZoneFixture
def test_zone_create_secondary_with_all_args(self):
    zone_name = random_zone_name()
    fixture = self.useFixture(ZoneFixture(name=zone_name, description='A secondary zone', type='SECONDARY', masters='127.0.0.1'))
    zone = fixture.zone
    self.assertEqual(zone_name, zone.name)
    self.assertEqual('A secondary zone', zone.description)
    self.assertEqual('SECONDARY', zone.type)
    self.assertEqual('127.0.0.1', zone.masters)