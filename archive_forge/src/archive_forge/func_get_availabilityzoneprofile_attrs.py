import functools
import ipaddress
from openstackclient.identity import common as identity_common
from osc_lib import exceptions as osc_exc
from osc_lib import utils
from oslo_utils import uuidutils
from octaviaclient.api import exceptions
from octaviaclient.osc.v2 import constants
def get_availabilityzoneprofile_attrs(client_manager, parsed_args):
    attr_map = {'name': ('name', str), 'availabilityzoneprofile': ('availability_zone_profile_id', 'availability_zone_profiles', client_manager.load_balancer.availabilityzoneprofile_list), 'provider': ('provider_name', str), 'availability_zone_data': ('availability_zone_data', str)}
    _attrs = vars(parsed_args)
    attrs = _map_attrs(_attrs, attr_map)
    return attrs