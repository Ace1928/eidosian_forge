from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.datagen import random_blacklist
from designateclient.functionaltests.v2.fixtures import BlacklistFixture
def test_zone_blacklist_create_and_show(self):
    client = self.clients.as_user('admin')
    blacklist = client.zone_blacklist_show(self.blacklist.id)
    self.assertEqual(self.blacklist.created_at, blacklist.created_at)
    self.assertEqual(self.blacklist.description, blacklist.description)
    self.assertEqual(self.blacklist.id, blacklist.id)
    self.assertEqual(self.blacklist.pattern, blacklist.pattern)
    self.assertEqual(self.blacklist.updated_at, blacklist.updated_at)