import copy
import testtools
from ironicclient.tests.unit import utils
import ironicclient.v1.portgroup
def test_portgroups_list_by_address(self):
    portgroups = self.mgr.list(address=PORTGROUP['address'])
    expect = [('GET', '/v1/portgroups/?address=%s' % PORTGROUP['address'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(1, len(portgroups))
    expected_resp = ({}, {'portgroups': [PORTGROUP, PORTGROUP2]})
    self.assertEqual(expected_resp, self.api.responses['/v1/portgroups']['GET'])