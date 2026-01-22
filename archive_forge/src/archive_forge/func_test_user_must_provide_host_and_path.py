import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlparse, parse_qsl, assertRaisesRegex
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import Provider
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.providers import get_driver
from libcloud.loadbalancer.drivers.cloudstack import CloudStackLBDriver
def test_user_must_provide_host_and_path(self):
    CloudStackLBDriver.path = None
    CloudStackLBDriver.type = Provider.CLOUDSTACK
    expected_msg = 'When instantiating CloudStack driver directly ' + 'you also need to provide host and path argument'
    cls = get_driver(Provider.CLOUDSTACK)
    assertRaisesRegex(self, Exception, expected_msg, cls, 'key', 'secret')
    try:
        cls('key', 'secret', True, 'localhost', '/path')
    except Exception:
        self.fail('host and path provided but driver raised an exception')