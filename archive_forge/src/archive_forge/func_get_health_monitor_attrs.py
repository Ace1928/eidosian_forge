import functools
import ipaddress
from openstackclient.identity import common as identity_common
from osc_lib import exceptions as osc_exc
from osc_lib import utils
from oslo_utils import uuidutils
from octaviaclient.api import exceptions
from octaviaclient.osc.v2 import constants
def get_health_monitor_attrs(client_manager, parsed_args):
    attr_map = {'health_monitor': ('health_monitor_id', 'healthmonitors', client_manager.load_balancer.health_monitor_list), 'project': ('project_id', 'project', client_manager.identity), 'name': ('name', str), 'pool': ('pool_id', 'pools', client_manager.load_balancer.pool_list), 'delay': ('delay', int), 'expected_codes': ('expected_codes', str), 'max_retries': ('max_retries', int), 'http_method': ('http_method', str), 'type': ('type', str), 'timeout': ('timeout', int), 'max_retries_down': ('max_retries_down', int), 'url_path': ('url_path', str), 'enable': ('admin_state_up', lambda x: True), 'disable': ('admin_state_up', lambda x: False), 'http_version': ('http_version', float), 'domain_name': ('domain_name', str)}
    add_tags_attr_map(attr_map)
    _attrs = vars(parsed_args)
    attrs = _map_attrs(_attrs, attr_map)
    return attrs