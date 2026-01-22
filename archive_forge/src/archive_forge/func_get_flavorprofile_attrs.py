import functools
import ipaddress
from openstackclient.identity import common as identity_common
from osc_lib import exceptions as osc_exc
from osc_lib import utils
from oslo_utils import uuidutils
from octaviaclient.api import exceptions
from octaviaclient.osc.v2 import constants
def get_flavorprofile_attrs(client_manager, parsed_args):
    attr_map = {'name': ('name', str), 'flavorprofile': ('flavorprofile_id', 'flavorprofiles', client_manager.load_balancer.flavorprofile_list), 'provider': ('provider_name', str), 'flavor_data': ('flavor_data', str)}
    _attrs = vars(parsed_args)
    attrs = _map_attrs(_attrs, attr_map)
    return attrs