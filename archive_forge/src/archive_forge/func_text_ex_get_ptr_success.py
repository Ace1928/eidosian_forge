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
def text_ex_get_ptr_success(self):
    service_name = 'cloudServersOpenStack'
    records = self.driver.ex_iterate_ptr_records(service_name)
    original = next(records)
    found = self.driver.ex_get_ptr_record(service_name, original.id)
    for attr in dir(original):
        self.assertEqual(getattr(found, attr), getattr(original, attr))