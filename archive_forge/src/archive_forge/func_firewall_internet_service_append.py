from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def firewall_internet_service_append(data, fos):
    vdom = data['vdom']
    firewall_internet_service_append_data = data['firewall_internet_service_append']
    filtered_data = underscore_to_hyphen(filter_firewall_internet_service_append_data(firewall_internet_service_append_data))
    return fos.set('firewall', 'internet-service-append', data=filtered_data, vdom=vdom)