import sys
import json
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.dns.base import Zone
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.common.types import ProviderError
from libcloud.test.secrets import DNS_PARAMS_AURORADNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.auroradns import AuroraDNSDriver, AuroraDNSHealthCheckType
def test_merge_extra_data(self):
    rdata = {'name': 'localhost', 'type': RecordType.A, 'content': '127.0.0.1'}
    params = {'ttl': 900, 'prio': 0, 'health_check_id': None, 'disabled': False}
    for param in params:
        extra = {param: params[param]}
        data = self.driver._AuroraDNSDriver__merge_extra_data(rdata, extra)
        self.assertEqual(data['content'], '127.0.0.1')
        self.assertEqual(data['type'], RecordType.A)
        self.assertEqual(data[param], params[param])
        self.assertEqual(data['name'], 'localhost')