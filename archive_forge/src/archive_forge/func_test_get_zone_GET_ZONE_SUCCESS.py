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
def test_get_zone_GET_ZONE_SUCCESS(self):
    ZonomiMockHttp.type = 'GET_ZONE_SUCCESS'
    zone = self.driver.get_zone(zone_id='gamertest.com')
    self.assertEqual(zone.id, 'gamertest.com')
    self.assertEqual(zone.domain, 'gamertest.com')
    self.assertEqual(zone.type, 'master')
    self.assertIsNone(zone.ttl)
    self.assertEqual(zone.driver, self.driver)