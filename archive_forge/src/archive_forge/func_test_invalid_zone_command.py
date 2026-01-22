from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.datagen import random_zone_name
from designateclient.functionaltests.v2.fixtures import ZoneFixture
def test_invalid_zone_command(self):
    cmd = 'zone hopefullynotacommand'
    self.assertRaises(CommandFailed, self.clients.openstack, cmd)