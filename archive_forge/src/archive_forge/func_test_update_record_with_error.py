import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_POINTDNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.pointdns import PointDNSDriver, PointDNSException
def test_update_record_with_error(self):
    PointDNSMockHttp.type = 'GET'
    record = self.driver.get_record(zone_id='1', record_id='141')
    PointDNSMockHttp.type = 'UPDATE_RECORD_WITH_ERROR'
    extra = {'ttl': 4500}
    try:
        self.driver.update_record(record=record, name='updated.com', type=RecordType.A, data='1.2.3.5', extra=extra)
    except PointDNSException:
        pass
    else:
        self.fail('Exception was not thrown')