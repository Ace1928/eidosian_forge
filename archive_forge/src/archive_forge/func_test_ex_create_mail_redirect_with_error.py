import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_POINTDNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.pointdns import PointDNSDriver, PointDNSException
def test_ex_create_mail_redirect_with_error(self):
    PointDNSMockHttp.type = 'GET'
    zone = self.driver.list_zones()[0]
    PointDNSMockHttp.type = 'CREATE_WITH_ERROR'
    try:
        self.driver.ex_create_mail_redirect('user@example-site.com', 'admin', zone)
    except PointDNSException as e:
        self.assertEqual(e.http_code, httplib.METHOD_NOT_ALLOWED)
    else:
        self.fail('Exception was not thrown')