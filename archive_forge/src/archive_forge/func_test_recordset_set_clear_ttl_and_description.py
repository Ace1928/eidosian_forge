from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.datagen import random_a_recordset_name
from designateclient.functionaltests.datagen import random_zone_name
from designateclient.functionaltests.v2.fixtures import RecordsetFixture
from designateclient.functionaltests.v2.fixtures import ZoneFixture
def test_recordset_set_clear_ttl_and_description(self):
    rset = self.clients.recordset_set(self.zone.id, self.recordset.id, no_description=True, no_ttl=True)
    self.assertEqual(rset.description, 'None')
    self.assertEqual(rset.ttl, 'None')