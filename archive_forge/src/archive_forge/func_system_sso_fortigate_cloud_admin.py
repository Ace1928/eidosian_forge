from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def system_sso_fortigate_cloud_admin(data, fos):
    vdom = data['vdom']
    state = data['state']
    system_sso_fortigate_cloud_admin_data = data['system_sso_fortigate_cloud_admin']
    filtered_data = underscore_to_hyphen(filter_system_sso_fortigate_cloud_admin_data(system_sso_fortigate_cloud_admin_data))
    if state == 'present' or state is True:
        return fos.set('system', 'sso-fortigate-cloud-admin', data=filtered_data, vdom=vdom)
    elif state == 'absent':
        return fos.delete('system', 'sso-fortigate-cloud-admin', mkey=filtered_data['name'], vdom=vdom)
    else:
        fos._module.fail_json(msg='state must be present or absent!')