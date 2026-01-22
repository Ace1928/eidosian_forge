import sys
import json
from libcloud.test import MockHttp, unittest
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib, urlparse
from libcloud.common.types import LibcloudError
from libcloud.test.secrets import DNS_PARAMS_CLOUDFLARE
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.cloudflare import (
def test_update_record_with_property_that_cant_be_updated(self):
    zone = self.driver.list_zones()[0]
    record = zone.list_records()[0]
    updated_record = self.driver.update_record(record=record, data='127.0.0.4', extra={'locked': True})
    self.assertNotEqual(updated_record.extra['locked'], True)