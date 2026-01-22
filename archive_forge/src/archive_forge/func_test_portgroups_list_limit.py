import copy
import testtools
from ironicclient.tests.unit import utils
import ironicclient.v1.portgroup
def test_portgroups_list_limit(self):
    portgroups = self.mgr.list(limit=1)
    expect = [('GET', '/v1/portgroups/?limit=1', {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(1, len(portgroups))
    expected_resp = ({}, {'next': 'http://127.0.0.1:6385/v1/portgroups/?limit=1', 'portgroups': [PORTGROUP]})
    self.assertEqual(expected_resp, self.api.responses['/v1/portgroups']['GET'])