from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.client import DesignateCLI
from designateclient.functionaltests.datagen import random_zone_name
from designateclient.functionaltests.v2.fixtures import SharedZoneFixture
from designateclient.functionaltests.v2.fixtures import ZoneFixture
def test_unshare_zone(self):
    shared_zone = self.useFixture(SharedZoneFixture(zone_id=self.zone.id, target_tenant_id=self.target_client.project_id)).zone_share
    shared_zones = self.clients.shared_zone_list(self.zone.id)
    self.assertTrue(self._is_entity_in_list(shared_zone, shared_zones))
    self.clients.unshare_zone(self.zone.id, shared_zone.id)
    shared_zones = self.clients.shared_zone_list(self.zone.id)
    self.assertFalse(self._is_entity_in_list(shared_zone, shared_zones))