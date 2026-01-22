import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.dns.drivers.nfsn import NFSNDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
def test_delete_record_not_found(self):
    NFSNMockHttp.type = 'NOT_FOUND'
    with self.assertRaises(RecordDoesNotExistError):
        self.assertTrue(self.test_record.delete())