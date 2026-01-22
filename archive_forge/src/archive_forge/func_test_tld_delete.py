from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.datagen import random_tld
from designateclient.functionaltests.v2.fixtures import TLDFixture
def test_tld_delete(self):
    client = self.clients.as_user('admin')
    client.tld_delete(self.tld.id)
    self.assertRaises(CommandFailed, client.tld_show, self.tld.id)