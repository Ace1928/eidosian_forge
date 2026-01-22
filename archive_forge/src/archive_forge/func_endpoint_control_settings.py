from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def endpoint_control_settings(data, fos):
    vdom = data['vdom']
    endpoint_control_settings_data = data['endpoint_control_settings']
    filtered_data = underscore_to_hyphen(filter_endpoint_control_settings_data(endpoint_control_settings_data))
    return fos.set('endpoint-control', 'settings', data=filtered_data, vdom=vdom)