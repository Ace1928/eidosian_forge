import sys
import json
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, ZoneAlreadyExistsError
from libcloud.utils.py3 import httplib
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.powerdns import PowerDNSDriver
def test_get_missing_zone(self):
    PowerDNSMockHttp.type = 'MISSING'
    with self.assertRaises(ZoneDoesNotExistError):
        self.driver.get_zone('example.com.')