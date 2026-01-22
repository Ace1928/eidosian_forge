import sys
import unittest
from unittest.mock import MagicMock
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_ZONOMI
from libcloud.dns.drivers.zonomi import ZonomiDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
def test_get_zone_GET_ZONE_DOES_NOT_EXIST(self):
    ZonomiMockHttp.type = 'GET_ZONE_DOES_NOT_EXIST'
    try:
        self.driver.get_zone('testzone.com')
    except ZoneDoesNotExistError as e:
        self.assertEqual(e.zone_id, 'testzone.com')
    else:
        self.fail('Exception was not thrown.')