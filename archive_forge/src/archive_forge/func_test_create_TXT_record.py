import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_ROUTE53
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.route53 import Route53DNSDriver
def test_create_TXT_record(self):
    """
        Check that TXT records are created in quotes
        """
    zone = self.driver.list_zones()[0]
    record = self.driver.create_record(name='', zone=zone, type=RecordType.TXT, data='test')
    self.assertEqual(record.id, 'TXT:')
    self.assertEqual(record.name, '')
    self.assertEqual(record.zone, zone)
    self.assertEqual(record.type, RecordType.TXT)
    self.assertEqual(record.data, '"test"')