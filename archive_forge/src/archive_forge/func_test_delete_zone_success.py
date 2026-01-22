import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_LIQUIDWEB
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.liquidweb import LiquidWebDNSDriver
def test_delete_zone_success(self):
    LiquidWebMockHttp.type = 'DELETE_ZONE_SUCCESS'
    zone = self.test_zone
    status = self.driver.delete_zone(zone=zone)
    self.assertEqual(status, True)