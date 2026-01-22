import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_WORLDWIDEDNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.common.worldwidedns import InvalidDomainName, NonExistentDomain
from libcloud.dns.drivers.worldwidedns import WorldWideDNSError, WorldWideDNSDriver
def test_create_record_max_entry_reached_give_entry(self):
    WorldWideDNSMockHttp.type = 'CREATE_RECORD_MAX_ENTRIES'
    zone = self.driver.list_zones()[0]
    record = self.driver.get_record(zone.id, '23')
    self.assertEqual(record.id, '23')
    self.assertEqual(record.name, 'domain23')
    self.assertEqual(record.type, 'A')
    self.assertEqual(record.data, '0.0.0.23')
    WorldWideDNSMockHttp.type = 'CREATE_RECORD_MAX_ENTRIES_WITH_ENTRY'
    record = self.driver.create_record(name='domain23b', zone=zone, type=RecordType.A, data='0.0.0.41', extra={'entry': 23})
    zone = record.zone
    self.assertEqual(record.id, '23')
    self.assertEqual(record.name, 'domain23b')
    self.assertEqual(record.type, 'A')
    self.assertEqual(record.data, '0.0.0.41')