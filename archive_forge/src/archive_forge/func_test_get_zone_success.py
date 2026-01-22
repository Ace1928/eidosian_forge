import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_LIQUIDWEB
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.liquidweb import LiquidWebDNSDriver
def test_get_zone_success(self):
    LiquidWebMockHttp.type = 'GET_ZONE_SUCCESS'
    zone = self.driver.get_zone(zone_id='13')
    self.assertEqual(zone.id, '13')
    self.assertEqual(zone.domain, 'blogtest.com')
    self.assertEqual(zone.type, 'NATIVE')
    self.assertIsNone(zone.ttl)
    self.assertEqual(zone.driver, self.driver)