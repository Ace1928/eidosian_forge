from openstack.network.v2 import bgp_peer as _bgp_peer
from openstack.network.v2 import bgp_speaker as _bgp_speaker
from openstack.tests.functional import base
def test_get_bgp_speaker(self):
    sot = self.operator_cloud.network.get_bgp_speaker(self.SPEAKER.id)
    self.assertEqual(self.IP_VERSION, sot.ip_version)
    self.assertEqual(self.LOCAL_AS, sot.local_as)