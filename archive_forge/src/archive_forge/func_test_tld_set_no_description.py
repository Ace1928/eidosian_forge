from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.datagen import random_tld
from designateclient.functionaltests.v2.fixtures import TLDFixture
def test_tld_set_no_description(self):
    client = self.clients.as_user('admin')
    tld = client.tld_set(self.tld.id, no_description=True)
    self.assertEqual(tld.description, 'None')