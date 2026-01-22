import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.types import LibcloudError
from libcloud.compute.base import Node
from libcloud.test.secrets import DNS_PARAMS_RACKSPACE
from libcloud.loadbalancer.base import LoadBalancer
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.rackspace import RackspaceDNSDriver, RackspacePTRRecord
def test_force_auth_token_kwargs(self):
    kwargs = {'ex_force_auth_token': 'some-auth-token', 'ex_force_base_url': 'https://dns.api.rackspacecloud.com/v1.0/11111'}
    driver = self.klass(*DNS_PARAMS_RACKSPACE, **kwargs)
    driver.list_zones()
    self.assertEqual(kwargs['ex_force_auth_token'], driver.connection.auth_token)
    self.assertEqual('/v1.0/11111', driver.connection.request_path)