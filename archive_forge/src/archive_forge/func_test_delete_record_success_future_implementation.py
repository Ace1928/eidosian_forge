import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_DNSIMPLE
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.dnsimple import DNSimpleDNSDriver
def test_delete_record_success_future_implementation(self):
    zone = self.driver.list_zones()[0]
    records = self.driver.list_records(zone=zone)
    self.assertEqual(len(records), 3)
    record = records[1]
    DNSimpleDNSMockHttp.type = 'DELETE_204'
    status = self.driver.delete_record(record=record)
    self.assertTrue(status)