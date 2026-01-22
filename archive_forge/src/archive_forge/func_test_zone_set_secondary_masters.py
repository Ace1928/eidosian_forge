from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.datagen import random_zone_name
from designateclient.functionaltests.v2.fixtures import ZoneFixture
def test_zone_set_secondary_masters(self):
    fixture = self.useFixture(ZoneFixture(name=random_zone_name(), description='A secondary zone', type='SECONDARY', masters='127.0.0.1'))
    zone = fixture.zone
    self.assertEqual('127.0.0.1', zone.masters)
    zone = self.clients.zone_set(zone.id, masters='127.0.0.2')
    self.assertEqual('127.0.0.2', zone.masters)