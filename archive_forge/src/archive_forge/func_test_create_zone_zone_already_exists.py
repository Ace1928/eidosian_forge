import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.common.nsone import NsOneException
from libcloud.test.secrets import DNS_PARAMS_NSONE
from libcloud.dns.drivers.nsone import NsOneDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
def test_create_zone_zone_already_exists(self):
    NsOneMockHttp.type = 'CREATE_ZONE_ZONE_ALREADY_EXISTS'
    try:
        self.driver.create_zone(domain='newzone.com')
    except ZoneAlreadyExistsError as e:
        self.assertEqual(e.zone_id, 'newzone.com')
    else:
        self.fail('Exception was not thrown')