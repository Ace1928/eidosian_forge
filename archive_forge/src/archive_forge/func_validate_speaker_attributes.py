from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.common import validators
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.bgp import peer as bgp_peer
def validate_speaker_attributes(parsed_args):
    validators.validate_int_range(parsed_args, 'local_as', MIN_AS_NUM, MAX_AS_NUM)