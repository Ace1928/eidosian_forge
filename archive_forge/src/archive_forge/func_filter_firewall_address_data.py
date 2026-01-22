from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.comparison import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.comparison import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.comparison import (
def filter_firewall_address_data(json):
    option_list = ['allow_routing', 'associated_interface', 'cache_ttl', 'clearpass_spt', 'color', 'comment', 'country', 'end_ip', 'end_mac', 'epg_name', 'fabric_object', 'filter', 'fqdn', 'fsso_group', 'hw_model', 'hw_vendor', 'interface', 'list', 'macaddr', 'name', 'node_ip_only', 'obj_id', 'obj_tag', 'obj_type', 'organization', 'os', 'policy_group', 'route_tag', 'sdn', 'sdn_addr_type', 'sdn_tag', 'start_ip', 'start_mac', 'sub_type', 'subnet', 'subnet_name', 'sw_version', 'tag_detection_level', 'tag_type', 'tagging', 'tenant', 'type', 'uuid', 'visibility', 'wildcard', 'wildcard_fqdn']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary