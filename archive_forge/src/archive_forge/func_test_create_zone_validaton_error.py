import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_LINODE, DNS_KEYWORD_PARAMS_LINODE
from libcloud.common.linode import LinodeException
from libcloud.dns.drivers.linode import LinodeDNSDriver, LinodeDNSDriverV3
from libcloud.test.file_fixtures import DNSFileFixtures
def test_create_zone_validaton_error(self):
    LinodeMockHttp.type = 'VALIDATION_ERROR'
    try:
        self.driver.create_zone(domain='foo.bar.com', type='master', ttl=None, extra=None)
    except LinodeException:
        pass
    else:
        self.fail('Exception was not thrown')