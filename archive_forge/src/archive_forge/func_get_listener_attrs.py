import functools
import ipaddress
from openstackclient.identity import common as identity_common
from osc_lib import exceptions as osc_exc
from osc_lib import utils
from oslo_utils import uuidutils
from octaviaclient.api import exceptions
from octaviaclient.osc.v2 import constants
def get_listener_attrs(client_manager, parsed_args):
    attr_map = {'name': ('name', str), 'description': ('description', str), 'protocol': ('protocol', str), 'listener': ('listener_id', 'listeners', client_manager.load_balancer.listener_list), 'loadbalancer': ('loadbalancer_id', 'loadbalancers', client_manager.load_balancer.load_balancer_list), 'connection_limit': ('connection_limit', str), 'protocol_port': ('protocol_port', int), 'default_pool': ('default_pool_id', 'pools', client_manager.load_balancer.pool_list), 'project': ('project_id', 'project', client_manager.identity), 'enable': ('admin_state_up', lambda x: True), 'disable': ('admin_state_up', lambda x: False), 'insert_headers': ('insert_headers', _format_kv), 'default_tls_container_ref': ('default_tls_container_ref', str), 'sni_container_refs': ('sni_container_refs', list), 'timeout_client_data': ('timeout_client_data', int), 'timeout_member_connect': ('timeout_member_connect', int), 'timeout_member_data': ('timeout_member_data', int), 'timeout_tcp_inspect': ('timeout_tcp_inspect', int), 'client_ca_tls_container_ref': ('client_ca_tls_container_ref', _format_str_if_need_treat_unset), 'client_authentication': ('client_authentication', str), 'client_crl_container_ref': ('client_crl_container_ref', _format_str_if_need_treat_unset), 'allowed_cidrs': ('allowed_cidrs', list), 'tls_ciphers': ('tls_ciphers', str), 'tls_versions': ('tls_versions', list), 'alpn_protocols': ('alpn_protocols', list), 'hsts_max_age': ('hsts_max_age', int), 'hsts_include_subdomains': ('hsts_include_subdomains', bool), 'hsts_preload': ('hsts_preload', bool)}
    add_tags_attr_map(attr_map)
    _attrs = vars(parsed_args)
    attrs = _map_attrs(_attrs, attr_map)
    return attrs