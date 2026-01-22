import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.dns.drivers.nfsn import NFSNDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
def test_ex_get_records_by(self):
    NFSNMockHttp.type = 'ONE_RECORD'
    records = self.driver.ex_get_records_by(self.test_zone, type=RecordType.A)
    self.assertEqual(len(records), 1)
    record = records[0]
    self.assertEqual(record.name, '')
    self.assertEqual(record.data, '192.0.2.1')
    self.assertEqual(record.type, RecordType.A)
    self.assertEqual(record.ttl, 3600)