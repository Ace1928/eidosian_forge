import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_LIQUIDWEB
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.liquidweb import LiquidWebDNSDriver
def test_list_zones_success(self):
    zones = self.driver.list_zones()
    self.assertEqual(len(zones), 3)
    zone = zones[0]
    self.assertEqual(zone.id, '378451')
    self.assertEqual(zone.domain, 'blogtest.com')
    self.assertEqual(zone.type, 'NATIVE')
    self.assertEqual(zone.driver, self.driver)
    self.assertIsNone(zone.ttl)
    second_zone = zones[1]
    self.assertEqual(second_zone.id, '378449')
    self.assertEqual(second_zone.domain, 'oltjanotest.com')
    self.assertEqual(second_zone.type, 'NATIVE')
    self.assertEqual(second_zone.driver, self.driver)
    self.assertIsNone(second_zone.ttl)
    third_zone = zones[2]
    self.assertEqual(third_zone.id, '378450')
    self.assertEqual(third_zone.domain, 'pythontest.com')
    self.assertEqual(third_zone.type, 'NATIVE')
    self.assertEqual(third_zone.driver, self.driver)
    self.assertIsNone(third_zone.ttl)