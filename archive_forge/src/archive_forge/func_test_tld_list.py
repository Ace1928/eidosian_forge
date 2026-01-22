from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.datagen import random_tld
from designateclient.functionaltests.v2.fixtures import TLDFixture
def test_tld_list(self):
    tlds = self.clients.as_user('admin').tld_list()
    self.assertGreater(len(tlds), 0)