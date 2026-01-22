from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def extension_controller_fortigate(data, fos):
    vdom = data['vdom']
    state = data['state']
    extension_controller_fortigate_data = data['extension_controller_fortigate']
    filtered_data = underscore_to_hyphen(filter_extension_controller_fortigate_data(extension_controller_fortigate_data))
    if state == 'present' or state is True:
        return fos.set('extension-controller', 'fortigate', data=filtered_data, vdom=vdom)
    elif state == 'absent':
        return fos.delete('extension-controller', 'fortigate', mkey=filtered_data['name'], vdom=vdom)
    else:
        fos._module.fail_json(msg='state must be present or absent!')