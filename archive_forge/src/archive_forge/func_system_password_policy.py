from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def system_password_policy(data, fos):
    vdom = data['vdom']
    system_password_policy_data = data['system_password_policy']
    system_password_policy_data = flatten_multilists_attributes(system_password_policy_data)
    filtered_data = underscore_to_hyphen(filter_system_password_policy_data(system_password_policy_data))
    return fos.set('system', 'password-policy', data=filtered_data, vdom=vdom)