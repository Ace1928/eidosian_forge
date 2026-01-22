import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_POINTDNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.pointdns import PointDNSDriver, PointDNSException
def test_ex_delete_mail_redirect(self):
    PointDNSMockHttp.type = 'GET'
    zone = self.driver.list_zones()[0]
    mailredirect = self.driver.ex_get_mail_redirects(zone.id, '5')
    PointDNSMockHttp.type = 'DELETE'
    status = self.driver.ex_delete_mail_redirect(mailredirect)
    self.assertTrue(status)