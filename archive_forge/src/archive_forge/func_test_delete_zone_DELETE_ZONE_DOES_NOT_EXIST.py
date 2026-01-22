import sys
import unittest
from unittest.mock import MagicMock
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_ZONOMI
from libcloud.dns.drivers.zonomi import ZonomiDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
def test_delete_zone_DELETE_ZONE_DOES_NOT_EXIST(self):
    ZonomiMockHttp.type = 'DELETE_ZONE_DOES_NOT_EXIST'
    try:
        self.driver.delete_zone(zone=self.test_zone)
    except ZoneDoesNotExistError as e:
        self.assertEqual(e.zone_id, self.test_zone.id)
    else:
        self.fail('Exception was not thrown.')