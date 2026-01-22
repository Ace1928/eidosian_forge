import sys
import json
from libcloud.test import MockHttp, unittest
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib, urlparse
from libcloud.common.types import LibcloudError
from libcloud.test.secrets import DNS_PARAMS_CLOUDFLARE
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.cloudflare import (
def test_get_record(self):
    record = self.driver.get_record('1234', '364797364')
    self.assertEqual(record.id, '364797364')
    self.assertIsNone(record.name)
    self.assertEqual(record.type, 'A')
    self.assertEqual(record.data, '192.30.252.153')