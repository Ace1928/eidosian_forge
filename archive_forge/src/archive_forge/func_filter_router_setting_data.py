from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_router_setting_data(json):
    option_list = ['bgp_debug_flags', 'hostname', 'igmp_debug_flags', 'imi_debug_flags', 'isis_debug_flags', 'ospf_debug_events_flags', 'ospf_debug_ifsm_flags', 'ospf_debug_lsa_flags', 'ospf_debug_nfsm_flags', 'ospf_debug_nsm_flags', 'ospf_debug_packet_flags', 'ospf_debug_route_flags', 'ospf6_debug_events_flags', 'ospf6_debug_ifsm_flags', 'ospf6_debug_lsa_flags', 'ospf6_debug_nfsm_flags', 'ospf6_debug_nsm_flags', 'ospf6_debug_packet_flags', 'ospf6_debug_route_flags', 'pimdm_debug_flags', 'pimsm_debug_joinprune_flags', 'pimsm_debug_simple_flags', 'pimsm_debug_timer_flags', 'rip_debug_flags', 'ripng_debug_flags', 'show_filter']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary