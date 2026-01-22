from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.datagen import random_zone_file
from designateclient.functionaltests.v2.fixtures import ImportFixture
def test_list_zone_imports(self):
    zone_import = self.useFixture(ImportFixture(zone_file_contents=self.zone_file_contents)).zone_import
    zone_imports = self.clients.zone_import_list()
    self.assertGreater(len(zone_imports), 0)
    self.assertTrue(self._is_entity_in_list(zone_import, zone_imports))