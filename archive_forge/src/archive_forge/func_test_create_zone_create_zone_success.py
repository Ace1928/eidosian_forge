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
def test_create_zone_create_zone_success(self):
    ZonomiMockHttp.type = 'CREATE_ZONE_SUCCESS'
    zone = self.driver.create_zone(domain='myzone.com')
    self.assertEqual(zone.id, 'myzone.com')
    self.assertEqual(zone.domain, 'myzone.com')
    self.assertEqual(zone.type, 'master')
    self.assertIsNone(zone.ttl)