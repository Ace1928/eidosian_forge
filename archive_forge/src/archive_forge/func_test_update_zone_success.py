import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_LINODE, DNS_KEYWORD_PARAMS_LINODE
from libcloud.common.linode import LinodeException
from libcloud.dns.drivers.linode import LinodeDNSDriver, LinodeDNSDriverV3
from libcloud.test.file_fixtures import DNSFileFixtures
def test_update_zone_success(self):
    zone = self.driver.list_zones()[0]
    updated_zone = self.driver.update_zone(zone=zone, domain='libcloud.org', ttl=10, extra={'SOA_Email': 'bar@libcloud.org'})
    self.assertEqual(zone.extra['SOA_Email'], 'dns@example.com')
    self.assertEqual(updated_zone.id, zone.id)
    self.assertEqual(updated_zone.domain, 'libcloud.org')
    self.assertEqual(updated_zone.type, zone.type)
    self.assertEqual(updated_zone.ttl, 10)
    self.assertEqual(updated_zone.extra['SOA_Email'], 'bar@libcloud.org')
    self.assertEqual(updated_zone.extra['status'], zone.extra['status'])
    self.assertEqual(updated_zone.extra['description'], zone.extra['description'])