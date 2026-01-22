from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.datagen import random_tsigkey_name
from designateclient.functionaltests.datagen import random_tsigkey_secret
from designateclient.functionaltests.datagen import random_zone_name
from designateclient.functionaltests.v2.fixtures import TSIGKeyFixture
from designateclient.functionaltests.v2.fixtures import ZoneFixture
def test_tsigkey_create_invalid_flag(self):
    client = self.clients.as_user('admin')
    self.assertRaises(CommandFailed, client.openstack, 'tsigkey create --notanoption "junk"')