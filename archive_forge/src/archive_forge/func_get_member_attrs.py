import functools
import ipaddress
from openstackclient.identity import common as identity_common
from osc_lib import exceptions as osc_exc
from osc_lib import utils
from oslo_utils import uuidutils
from octaviaclient.api import exceptions
from octaviaclient.osc.v2 import constants
def get_member_attrs(client_manager, parsed_args):
    attr_map = {'name': ('name', str), 'address': ('address', str), 'protocol_port': ('protocol_port', int), 'project_id': ('project_id', 'project', client_manager.identity), 'pool': ('pool_id', 'pools', client_manager.load_balancer.pool_list), 'member': ('member_id', 'members', 'pool', client_manager.load_balancer.member_list), 'enable_backup': ('backup', lambda x: True), 'disable_backup': ('backup', lambda x: False), 'weight': ('weight', int), 'subnet_id': ('subnet_id', 'subnets', client_manager.neutronclient.list_subnets), 'monitor_port': ('monitor_port', int), 'monitor_address': ('monitor_address', str), 'enable': ('admin_state_up', lambda x: True), 'disable': ('admin_state_up', lambda x: False)}
    add_tags_attr_map(attr_map)
    _attrs = vars(parsed_args)
    attrs = _map_attrs(_attrs, attr_map)
    return attrs