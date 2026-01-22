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
def test_create_record_zone_not_found(self):
    NsOneMockHttp.type = 'CREATE_RECORD_ZONE_NOT_FOUND'
    try:
        self.driver.create_record(self.test_record.name, self.test_record.zone, self.test_record.type, self.test_record.data, self.test_record.extra)
    except NsOneException as err:
        self.assertEqual(err.message, 'zone not found')
    else:
        self.fail('Exception was not thrown')