from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def firewall_auth_portal(data, fos):
    vdom = data['vdom']
    firewall_auth_portal_data = data['firewall_auth_portal']
    filtered_data = underscore_to_hyphen(filter_firewall_auth_portal_data(firewall_auth_portal_data))
    return fos.set('firewall', 'auth-portal', data=filtered_data, vdom=vdom)