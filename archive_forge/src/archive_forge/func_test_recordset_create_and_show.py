from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.datagen import random_a_recordset_name
from designateclient.functionaltests.datagen import random_zone_name
from designateclient.functionaltests.v2.fixtures import RecordsetFixture
from designateclient.functionaltests.v2.fixtures import ZoneFixture
def test_recordset_create_and_show(self):
    rset = self.clients.recordset_show(self.zone.id, self.recordset.id)
    self.assertTrue(hasattr(self.recordset, 'action'))
    self.assertTrue(hasattr(rset, 'action'))
    self.assertEqual(self.recordset.created_at, rset.created_at)
    self.assertEqual(self.recordset.description, rset.description)
    self.assertEqual(self.recordset.id, rset.id)
    self.assertEqual(self.recordset.name, rset.name)
    self.assertEqual(self.recordset.records, rset.records)
    self.assertEqual(self.recordset.status, rset.status)
    self.assertEqual(self.recordset.ttl, rset.ttl)
    self.assertEqual(self.recordset.type, rset.type)
    self.assertEqual(self.recordset.updated_at, rset.updated_at)
    self.assertEqual(self.recordset.version, rset.version)
    self.assertEqual(self.recordset.zone_id, self.zone.id)