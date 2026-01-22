from openstack.network.v2 import bgp_peer as _bgp_peer
from openstack.network.v2 import bgp_speaker as _bgp_speaker
from openstack.tests.functional import base
def test_update_bgp_peer(self):
    name = 'new_peer_name' + self.getUniqueString()
    sot = self.operator_cloud.network.update_bgp_peer(self.PEER.id, name=name)
    self.assertEqual(name, sot.name)