import sys
import unittest
from unittest.mock import MagicMock
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_DURABLEDNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.durabledns import (
def test_delete_record_zone_does_not_exist(self):
    zone = self.driver.list_zones()[0]
    record = self.driver.list_records(zone=zone)[0]
    DurableDNSMockHttp.type = 'ZONE_DOES_NOT_EXIST'
    try:
        self.driver.delete_record(record=record)
    except ZoneDoesNotExistError:
        pass
    else:
        self.fail('Exception was not thrown')