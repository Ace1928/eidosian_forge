from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def firewall_ipv6_eh_filter(data, fos):
    vdom = data['vdom']
    firewall_ipv6_eh_filter_data = data['firewall_ipv6_eh_filter']
    firewall_ipv6_eh_filter_data = flatten_multilists_attributes(firewall_ipv6_eh_filter_data)
    filtered_data = underscore_to_hyphen(filter_firewall_ipv6_eh_filter_data(firewall_ipv6_eh_filter_data))
    return fos.set('firewall', 'ipv6-eh-filter', data=filtered_data, vdom=vdom)