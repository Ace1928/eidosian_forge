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
def test_create_zone_already_exists(self):
    ZonomiMockHttp.type = 'CREATE_ZONE_ALREADY_EXISTS'
    try:
        self.driver.create_zone(domain='gamertest.com')
    except ZoneAlreadyExistsError as e:
        self.assertEqual(e.zone_id, 'gamertest.com')
    else:
        self.fail('Exception was not thrown.')