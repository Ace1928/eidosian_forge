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
def test_update_record_zone_does_not_exist(self):
    z_extra = {'ns': 'ns1.durabledns.com.', 'mbox': 'mail.myzone.com', 'refresh': '13000', 'retry': 7200, 'expire': 1300, 'minimum': 13, 'xfer': '127.0.0.1', 'update_acl': '127.0.0.1'}
    zone = Zone(id='myzone.com.', domain='myzone.com.', type='master', ttl=1300, driver=self.driver, extra=z_extra)
    extra = {'aux': 1, 'ttl': 3600}
    record = Record(id='353286987', type='A', zone=zone, name='record1', data='192.168.0.1', driver=self, extra=extra)
    DurableDNSMockHttp.type = 'ZONE_DOES_NOT_EXIST'
    try:
        self.driver.update_record(record, record.name, record.type, record.data)
    except ZoneDoesNotExistError:
        pass
    else:
        self.fail('Exception was not thrown')