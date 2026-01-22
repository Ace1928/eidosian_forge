from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def log_memory_global_setting(data, fos):
    vdom = data['vdom']
    log_memory_global_setting_data = data['log_memory_global_setting']
    filtered_data = underscore_to_hyphen(filter_log_memory_global_setting_data(log_memory_global_setting_data))
    return fos.set('log.memory', 'global-setting', data=filtered_data, vdom=vdom)