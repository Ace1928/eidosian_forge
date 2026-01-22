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
def test_create_record_already_exists(self):
    NsOneMockHttp.type = 'CREATE_RECORD_ALREADY_EXISTS'
    try:
        self.driver.create_record(self.test_record.name, self.test_record.zone, self.test_record.type, self.test_record.data, self.test_record.extra)
    except RecordAlreadyExistsError as err:
        self.assertEqual(err.value, 'record already exists')
    else:
        self.fail('Exception was not thrown')