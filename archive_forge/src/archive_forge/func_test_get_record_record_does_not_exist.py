import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_LIQUIDWEB
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.liquidweb import LiquidWebDNSDriver
def test_get_record_record_does_not_exist(self):
    LiquidWebMockHttp.type = 'GET_RECORD_RECORD_DOES_NOT_EXIST'
    try:
        self.driver.get_record(zone_id='13', record_id='13')
    except RecordDoesNotExistError as e:
        self.assertEqual(e.record_id, '13')
    else:
        self.fail('Exception was not thrown')