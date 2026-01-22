import sys
import json
from libcloud.test import MockHttp, unittest
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib, urlparse
from libcloud.common.types import LibcloudError
from libcloud.test.secrets import DNS_PARAMS_CLOUDFLARE
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.cloudflare import (
def test_update_zone_with_property_that_cant_be_updated(self):
    zone = self.driver.list_zones()[0]
    updated_zone = self.driver.update_zone(zone, domain='', extra={'owner': 'owner'})
    self.assertEqual(zone, updated_zone)