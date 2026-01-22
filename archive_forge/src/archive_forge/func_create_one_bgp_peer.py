from unittest import mock
import uuid
from openstack.network.v2 import agent as _agent
from openstack.network.v2 import bgp_peer as _bgp_peer
from openstack.network.v2 import bgp_speaker as _bgp_speaker
from neutronclient.tests.unit.osc.v2 import fakes
@staticmethod
def create_one_bgp_peer(attrs=None):
    attrs = attrs or {}
    bgp_peer_attrs = {'auth_type': None, 'peer_ip': '1.1.1.1', 'remote_as': 100, 'id': uuid.uuid4().hex, 'name': 'bgp-peer-' + uuid.uuid4().hex, 'tenant_id': uuid.uuid4().hex}
    bgp_peer_attrs.update(attrs)
    ret_bgp_peer = _bgp_peer.BgpPeer(**bgp_peer_attrs)
    return ret_bgp_peer