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
def filter_firewall_service_custom_data(json):
    option_list = ['app_category', 'app_service_type', 'application', 'category', 'check_reset_range', 'color', 'comment', 'fabric_object', 'fqdn', 'helper', 'icmpcode', 'icmptype', 'iprange', 'name', 'protocol', 'protocol_number', 'proxy', 'sctp_portrange', 'session_ttl', 'tcp_halfclose_timer', 'tcp_halfopen_timer', 'tcp_portrange', 'tcp_rst_timer', 'tcp_timewait_timer', 'udp_idle_timer', 'udp_portrange', 'visibility']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary