import unittest
from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.client import DesignateCLI
from designateclient.functionaltests.datagen import random_zone_name
from designateclient.functionaltests.v2.fixtures import TransferRequestFixture
from designateclient.functionaltests.v2.fixtures import ZoneFixture
def test_zone_transfer_accept_request(self):
    self.target_client.zone_transfer_accept_request(id=self.transfer_request.id, key=self.transfer_request.key)
    self.target_client.zone_show(self.zone.id)
    self.assertRaises(CommandFailed, self.clients.zone_show, self.zone.id)