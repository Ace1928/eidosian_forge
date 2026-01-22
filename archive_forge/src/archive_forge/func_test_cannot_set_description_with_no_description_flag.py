from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.datagen import random_blacklist
from designateclient.functionaltests.v2.fixtures import BlacklistFixture
def test_cannot_set_description_with_no_description_flag(self):
    client = self.clients.as_user('admin')
    self.assertRaises(CommandFailed, client.zone_blacklist_set, self.blacklist.id, pattern=random_blacklist(), description='new description', no_description=True)