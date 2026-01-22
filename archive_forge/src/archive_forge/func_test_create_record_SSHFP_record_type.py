import sys
import json
from libcloud.test import MockHttp, unittest
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib, urlparse
from libcloud.common.types import LibcloudError
from libcloud.test.secrets import DNS_PARAMS_CLOUDFLARE
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.cloudflare import (
def test_create_record_SSHFP_record_type(self):
    zone = self.driver.list_zones()[0]
    CloudFlareMockHttp.type = 'sshfp_record_type'
    record = self.driver.create_record(name='test_sshfp', zone=zone, type=RecordType.SSHFP, data='2 1 ABCDEF12345')
    self.assertEqual(record.id, '200')
    self.assertEqual(record.name, 'test_sshfp')
    self.assertEqual(record.type, 'SSHFP')
    self.assertEqual(record.data, '2 1 ABCDEF12345')