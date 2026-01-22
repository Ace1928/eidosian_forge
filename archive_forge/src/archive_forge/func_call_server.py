from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.common import validators
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.bgp import peer as bgp_peer
def call_server(self, neutron_client, search_opts, parsed_args):
    _speaker_id = get_bgp_speaker_id(neutron_client, parsed_args.bgp_speaker)
    data = neutron_client.list_route_advertised_from_bgp_speaker(_speaker_id, **search_opts)
    return data