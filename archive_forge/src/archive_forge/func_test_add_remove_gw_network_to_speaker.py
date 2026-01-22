from openstack.network.v2 import bgp_peer as _bgp_peer
from openstack.network.v2 import bgp_speaker as _bgp_speaker
from openstack.tests.functional import base
def test_add_remove_gw_network_to_speaker(self):
    net_name = 'my_network' + self.getUniqueString()
    net = self.user_cloud.create_network(name=net_name)
    self.operator_cloud.network.add_gateway_network_to_speaker(self.SPEAKER.id, net.id)
    sot = self.operator_cloud.network.get_bgp_speaker(self.SPEAKER.id)
    self.assertEqual([net.id], sot.networks)
    self.operator_cloud.network.remove_gateway_network_from_speaker(self.SPEAKER.id, net.id)
    sot = self.operator_cloud.network.get_bgp_speaker(self.SPEAKER.id)
    self.assertEqual([], sot.networks)