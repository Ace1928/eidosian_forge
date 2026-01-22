from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def firewall_access_proxy6(data, fos):
    vdom = data['vdom']
    state = data['state']
    firewall_access_proxy6_data = data['firewall_access_proxy6']
    firewall_access_proxy6_data = flatten_multilists_attributes(firewall_access_proxy6_data)
    filtered_data = underscore_to_hyphen(filter_firewall_access_proxy6_data(firewall_access_proxy6_data))
    if state == 'present' or state is True:
        return fos.set('firewall', 'access-proxy6', data=filtered_data, vdom=vdom)
    elif state == 'absent':
        return fos.delete('firewall', 'access-proxy6', mkey=filtered_data['name'], vdom=vdom)
    else:
        fos._module.fail_json(msg='state must be present or absent!')