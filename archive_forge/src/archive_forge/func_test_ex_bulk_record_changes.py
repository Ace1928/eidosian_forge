import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_GOOGLE, DNS_KEYWORD_PARAMS_GOOGLE
from libcloud.common.google import GoogleBaseAuthConnection
from libcloud.dns.drivers.google import GoogleDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.test.common.test_google import GoogleTestCase, GoogleAuthMockHttp
def test_ex_bulk_record_changes(self):
    zone = self.driver.get_zone('example-com')
    records = self.driver.ex_bulk_record_changes(zone, {})
    self.assertEqual(records['additions'][0].name, 'foo.example.com.')
    self.assertEqual(records['additions'][0].type, 'A')
    self.assertEqual(records['deletions'][0].name, 'bar.example.com.')
    self.assertEqual(records['deletions'][0].type, 'A')