import sys
import unittest
from unittest.mock import MagicMock
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_DURABLEDNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.durabledns import (
def test_create_record_no_extra_param(self):
    zone = self.driver.list_zones()[0]
    DurableDNSMockHttp.type = 'NO_EXTRA_PARAMS'
    record = self.driver.create_record(name='record1', zone=zone, type=RecordType.A, data='1.2.3.4')
    self.assertEqual(record.id, '353367855')
    self.assertEqual(record.name, 'record1')
    self.assertEqual(record.zone, zone)
    self.assertEqual(record.type, RecordType.A)
    self.assertEqual(record.data, '1.2.3.4')
    self.assertEqual(record.extra.get('aux'), RECORD_EXTRA_PARAMS_DEFAULT_VALUES.get('aux'))
    self.assertEqual(record.extra.get('ttl'), RECORD_EXTRA_PARAMS_DEFAULT_VALUES.get('ttl'))