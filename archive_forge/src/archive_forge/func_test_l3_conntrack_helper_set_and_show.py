import uuid
from openstackclient.tests.functional.network.v2 import common
def test_l3_conntrack_helper_set_and_show(self):
    helper = {'helper': 'tftp', 'protocol': 'udp', 'port': 69}
    router_id = self._create_router()
    created_helper = self._create_helpers(router_id, [helper])[0]
    output = self.openstack('network l3 conntrack helper show %(router_id)s %(ct_id)s -f json' % {'router_id': router_id, 'ct_id': created_helper['id']}, parse_output=True)
    self.assertEqual(helper['helper'], output['helper'])
    self.assertEqual(helper['protocol'], output['protocol'])
    self.assertEqual(helper['port'], output['port'])
    raw_output = self.openstack('network l3 conntrack helper set %(router_id)s %(ct_id)s --port %(port)s ' % {'router_id': router_id, 'ct_id': created_helper['id'], 'port': helper['port'] + 1})
    self.assertOutput('', raw_output)
    output = self.openstack('network l3 conntrack helper show %(router_id)s %(ct_id)s -f json' % {'router_id': router_id, 'ct_id': created_helper['id']}, parse_output=True)
    self.assertEqual(helper['port'] + 1, output['port'])
    self.assertEqual(helper['helper'], output['helper'])
    self.assertEqual(helper['protocol'], output['protocol'])