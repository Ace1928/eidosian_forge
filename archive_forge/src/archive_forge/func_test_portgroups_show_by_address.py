import copy
import testtools
from ironicclient.tests.unit import utils
import ironicclient.v1.portgroup
def test_portgroups_show_by_address(self):
    portgroup = self.mgr.get_by_address(PORTGROUP['address'])
    expect = [('GET', '/v1/portgroups/detail?address=%s' % PORTGROUP['address'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(PORTGROUP['uuid'], portgroup.uuid)
    self.assertEqual(PORTGROUP['name'], portgroup.name)
    self.assertEqual(PORTGROUP['node_uuid'], portgroup.node_uuid)
    self.assertEqual(PORTGROUP['address'], portgroup.address)
    expected_resp = ({}, {'portgroups': [PORTGROUP]})
    self.assertEqual(expected_resp, self.api.responses['/v1/portgroups/detail?address=%s' % PORTGROUP['address']]['GET'])