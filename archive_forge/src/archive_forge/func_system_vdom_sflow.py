from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def system_vdom_sflow(data, fos):
    vdom = data['vdom']
    system_vdom_sflow_data = data['system_vdom_sflow']
    filtered_data = underscore_to_hyphen(filter_system_vdom_sflow_data(system_vdom_sflow_data))
    return fos.set('system', 'vdom-sflow', data=filtered_data, vdom=vdom)