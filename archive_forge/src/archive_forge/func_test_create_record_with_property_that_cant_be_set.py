import sys
import json
from libcloud.test import MockHttp, unittest
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib, urlparse
from libcloud.common.types import LibcloudError
from libcloud.test.secrets import DNS_PARAMS_CLOUDFLARE
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.cloudflare import (
def test_create_record_with_property_that_cant_be_set(self):
    zone = self.driver.list_zones()[0]
    record = self.driver.create_record(name='test5', zone=zone, type=RecordType.A, data='127.0.0.3', extra={'locked': True})
    self.assertNotEqual(record.extra['locked'], True)