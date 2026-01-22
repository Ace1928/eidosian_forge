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
def test_ex_list_ptr_success(self):
    records = self.driver.ex_iterate_ptr_records(RDNS_NODE)
    for record in records:
        self.assertTrue(isinstance(record, RackspacePTRRecord))
        self.assertEqual(record.type, RecordType.PTR)
        self.assertEqual(record.extra['uri'], RDNS_NODE.extra['uri'])
        self.assertEqual(record.extra['service_name'], RDNS_NODE.extra['service_name'])