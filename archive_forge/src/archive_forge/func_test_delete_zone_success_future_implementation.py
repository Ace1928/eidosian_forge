import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_DNSIMPLE
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.dnsimple import DNSimpleDNSDriver
def test_delete_zone_success_future_implementation(self):
    zone = self.driver.list_zones()[0]
    DNSimpleDNSMockHttp.type = 'DELETE_204'
    status = self.driver.delete_zone(zone=zone)
    self.assertTrue(status)