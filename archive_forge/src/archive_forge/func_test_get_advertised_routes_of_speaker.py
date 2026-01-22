from openstack.network.v2 import bgp_peer as _bgp_peer
from openstack.network.v2 import bgp_speaker as _bgp_speaker
from openstack.tests.functional import base
def test_get_advertised_routes_of_speaker(self):
    sot = self.operator_cloud.network.get_advertised_routes_of_speaker(self.SPEAKER.id)
    self.assertEqual({'advertised_routes': []}, sot)