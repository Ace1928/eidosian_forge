from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.common import validators
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.bgp import peer as bgp_peer
def get_bgp_speaker_id(client, id_or_name):
    return neutronv20.find_resourceid_by_name_or_id(client, 'bgp_speaker', id_or_name)