from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def router_rip(data, fos):
    vdom = data['vdom']
    router_rip_data = data['router_rip']
    router_rip_data = flatten_multilists_attributes(router_rip_data)
    filtered_data = underscore_to_hyphen(filter_router_rip_data(router_rip_data))
    return fos.set('router', 'rip', data=filtered_data, vdom=vdom)