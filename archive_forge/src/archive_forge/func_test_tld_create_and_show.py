from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.datagen import random_tld
from designateclient.functionaltests.v2.fixtures import TLDFixture
def test_tld_create_and_show(self):
    tld = self.clients.as_user('admin').tld_show(self.tld.id)
    self.assertEqual(tld.name, self.tld.name)
    self.assertEqual(tld.created_at, self.tld.created_at)
    self.assertEqual(tld.id, self.tld.id)
    self.assertEqual(tld.name, self.tld.name)
    self.assertEqual(tld.updated_at, self.tld.updated_at)