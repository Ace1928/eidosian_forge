import unittest
from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.client import DesignateCLI
from designateclient.functionaltests.datagen import random_zone_name
from designateclient.functionaltests.v2.fixtures import TransferRequestFixture
from designateclient.functionaltests.v2.fixtures import ZoneFixture
def test_create_and_show_zone_transfer_request(self):
    transfer_request = self.useFixture(TransferRequestFixture(zone=self.zone, user='default', target_user='alt')).transfer_request
    fetched_xfr = self.clients.zone_transfer_request_show(transfer_request.id)
    self.assertEqual(fetched_xfr.created_at, transfer_request.created_at)
    self.assertEqual(fetched_xfr.description, transfer_request.description)
    self.assertEqual(fetched_xfr.id, transfer_request.id)
    self.assertEqual(fetched_xfr.key, transfer_request.key)
    self.assertEqual(fetched_xfr.links, transfer_request.links)
    self.assertEqual(fetched_xfr.target_project_id, transfer_request.target_project_id)
    self.assertEqual(fetched_xfr.updated_at, transfer_request.updated_at)
    self.assertEqual(fetched_xfr.status, transfer_request.status)
    self.assertEqual(fetched_xfr.zone_id, self.zone.id)
    self.assertEqual(fetched_xfr.zone_name, self.zone.name)