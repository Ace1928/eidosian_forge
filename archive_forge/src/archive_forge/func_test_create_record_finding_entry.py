import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_WORLDWIDEDNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.common.worldwidedns import InvalidDomainName, NonExistentDomain
from libcloud.dns.drivers.worldwidedns import WorldWideDNSError, WorldWideDNSDriver
def test_create_record_finding_entry(self):
    zone = self.driver.list_zones()[0]
    WorldWideDNSMockHttp.type = 'CREATE_RECORD'
    record = self.driver.create_record(name='domain4', zone=zone, type=RecordType.A, data='0.0.0.4')
    WorldWideDNSMockHttp.type = 'CREATE_SECOND_RECORD'
    zone = record.zone
    record2 = self.driver.create_record(name='domain1', zone=zone, type=RecordType.A, data='0.0.0.1')
    self.assertEqual(record.id, '4')
    self.assertEqual(record2.id, '5')