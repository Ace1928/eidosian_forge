import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.test.secrets import DNS_PARAMS_ZERIGO
from libcloud.dns.drivers.zerigo import ZerigoError, ZerigoDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
def test_record_with_empty_name(self):
    zone = self.driver.list_zones()[0]
    record1 = list(self.driver.list_records(zone=zone))[-1]
    record2 = list(self.driver.list_records(zone=zone))[-2]
    self.assertIsNone(record1.name)
    self.assertIsNone(record2.name)