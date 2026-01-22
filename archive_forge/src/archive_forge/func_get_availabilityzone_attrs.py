import functools
import ipaddress
from openstackclient.identity import common as identity_common
from osc_lib import exceptions as osc_exc
from osc_lib import utils
from oslo_utils import uuidutils
from octaviaclient.api import exceptions
from octaviaclient.osc.v2 import constants
def get_availabilityzone_attrs(client_manager, parsed_args):
    attr_map = {'name': ('name', str), 'availabilityzone': ('availabilityzone_name', 'availability_zones', client_manager.load_balancer.availabilityzone_list), 'availabilityzoneprofile': ('availability_zone_profile_id', 'availability_zone_profiles', client_manager.load_balancer.availabilityzoneprofile_list), 'enable': ('enabled', lambda x: True), 'disable': ('enabled', lambda x: False), 'description': ('description', str)}
    _attrs = vars(parsed_args)
    attrs = _map_attrs(_attrs, attr_map)
    return attrs