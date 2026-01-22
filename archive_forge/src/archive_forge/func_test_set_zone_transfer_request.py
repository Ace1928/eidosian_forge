import unittest
from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.client import DesignateCLI
from designateclient.functionaltests.datagen import random_zone_name
from designateclient.functionaltests.v2.fixtures import TransferRequestFixture
from designateclient.functionaltests.v2.fixtures import ZoneFixture
@unittest.skip('Fails because `zone transfer request set` returns nothing')
def test_set_zone_transfer_request(self):
    transfer_request = self.useFixture(TransferRequestFixture(zone=self.zone, description='old description')).transfer_request
    self.assertEqual(transfer_request.description, 'old description')
    updated_xfr = self.clients.zone_transfer_request_set(transfer_request.id, description='updated description')
    self.assertEqual(updated_xfr.description, 'updated description')